#ifndef KET_MPI_STATE_HPP
# define KET_MPI_STATE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <iterator>
# include <vector>
# include <memory>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <array>
# include <initializer_list>

# include <boost/iterator/iterator_facade.hpp>

# include <boost/range/sub_range.hpp>
# include <boost/range/iterator_range.hpp>
# include <boost/range/iterator.hpp>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/iterator_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/transform_inclusive_scan.hpp>
# include <ket/mpi/utility/transform_inclusive_scan_self.hpp>
# include <ket/mpi/utility/upper_bound.hpp>
# include <ket/mpi/utility/detail/swap_local_qubits.hpp>

# if __cplusplus >= 201703L
#   define KET_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define KET_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace ket
{
  namespace mpi
  {
    template <
      typename Complex, int num_page_qubits = 0,
      typename Allocator = std::allocator<Complex> >
    class state;


    namespace state_detail
    {
      template <typename State>
      class state_iterator
        : public boost::iterators::iterator_facade<
            state_iterator<State>,
            typename State::value_type,
            std::random_access_iterator_tag>
      {
        State* state_ptr_;
        int index_;

        friend class boost::iterators::iterator_core_access;

       public:
        constexpr state_iterator() noexcept
          : state_ptr_{nullptr}, index_{}
        { }

        constexpr state_iterator(State& state, int const index) noexcept
          : state_ptr_{std::addressof(state)}, index_{index}
        { }


        typename State::value_type& dereference() const
        { return const_cast<typename State::value_type&>((*state_ptr_)[index_]); }

        bool equal(state_iterator const& other) const
        { return state_ptr_ == other.state_ptr_ and index_ == other.index_; }

        state_iterator& increment() { ++index_; return *this; }
        state_iterator& decrement() { --index_; return *this; }
        state_iterator& advance(typename State::difference_type const n)
        { index_ += n; return *this; }
        typename State::difference_type distance_to(state_iterator const& other) const
        { return other.index_-index_; }

        void swap(state_iterator& other) noexcept
        {
          using std::swap;
          swap(state_ptr_, other.state_ptr_);
          swap(index_, other.index_);
        }
      }; // class state_iterator<State>

      template <typename State>
      inline void swap(
        ::ket::mpi::state_detail::state_iterator<State>& lhs,
        ::ket::mpi::state_detail::state_iterator<State>& rhs) noexcept
      { lhs.swap(rhs); }
    } // namespace state_detail


    // NOTE: Assuming size() % (1 << num_page_qubits) == 0
    template <typename Complex, int num_page_qubits, typename Allocator>
    class state
    {
     public:
      static constexpr std::size_t num_pages = 1u << num_page_qubits;

      using value_type = Complex;
      using allocator_type = typename Allocator::template rebind<value_type>::other;

     private:
      using data_type = std::vector<value_type, allocator_type>;
      data_type data_;

     public:
      using page_range_type
        = boost::iterator_range<typename ::ket::utility::meta::iterator_of<data_type>::type>;

     public:
      std::size_t num_local_qubits_;
      std::array<page_range_type, num_pages> page_ranges_;
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

      state() noexcept(noexcept(allocator_type{}))
        : data_{allocator_type{}}, num_local_qubits_{}, page_ranges_{}, buffer_range_{}
      { }

      explicit state(allocator_type const& allocator) noexcept
        : data_{allocator}, num_local_qubits_{}, page_ranges_{}, buffer_range_{}
      { }

      ~state() noexcept = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
      state(state&&) = default;
      state& operator=(state&&) = default;

      state(state const& other, allocator_type const& allocator)
        : data_{other.data_, allocator},
          num_local_qubits_{other.num_local_qubits_},
          page_ranges_{other.page_ranges_},
          buffer_range_{other.buffer_range_}
      { }

      state(state&& other, allocator_type const& allocator)
        : data_{std::move(other.data_), allocator},
          num_local_qubits_{std::move(other.num_local_qubits_)},
          page_ranges_{std::move(other.page_ranges_)},
          buffer_range_{std::move(other.buffer_range_)}
      { }

      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_{generate_initial_data(initializer_list, allocator)},
          num_local_qubits_{::ket::utility::integer_log2(initializer_list.size())},
          page_ranges_{generate_initial_page_ranges(data_)},
          buffer_range_{generate_initial_buffer_range(data_)}
      { }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_{generate_initial_data(
            ::ket::mpi::utility::policy::make_general_mpi(),
            num_local_qubits, initial_integer, permutation, communicator, environment)},
          num_local_qubits_{num_local_qubits},
          page_ranges_{generate_initial_page_ranges(data_)},
          buffer_range_{generate_initial_buffer_range(data_)}
      { }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_{generate_initial_data(
            mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment)},
          num_local_qubits_{num_local_qubits},
          page_ranges_{generate_initial_page_ranges(data_)},
          buffer_range_{generate_initial_buffer_range(data_)}
      { }

      void swap_pages(size_type const page_id1, size_type const page_id2)
      {
        assert(page_id1 < num_pages);
        assert(page_id2 < num_pages);
        assert(page_id1 != page_id2);

        using std::swap;
        swap(page_ranges_[page_id1], page_ranges_[page_id2]);
      }

      void swap_buffer_and_page(size_type const page_id)
      {
        assert(page_id < num_pages);

        using std::swap;
        swap(buffer_range_, page_ranges_[page_id]);
      }

      void swap_values(
        std::pair<size_type, size_type> const& page_nonpage_index_pair1,
        std::pair<size_type, size_type> const& page_nonpage_index_pair2)
      {
        assert(page_nonpage_index_pair1.first < num_pages);
        assert(page_nonpage_index_pair2.first < num_pages);
        assert(page_nonpage_index_pair1.first != page_nonpage_index_pair2.first);

        using std::swap;
        swap(
          ::ket::utility::begin(page_ranges_[page_nonpage_index_pair1.first])[page_nonpage_index_pair1.second],
          ::ket::utility::begin(page_ranges_[page_nonpage_index_pair2.first])[page_nonpage_index_pair2.second]);
      }

      page_range_type& page_range(size_type const page_id)
      { return page_ranges_[page_id]; }

      page_range_type const& page_range(size_type const page_id) const
      { return page_ranges_[page_id]; }

      page_range_type& buffer_range()
      { return buffer_range_; }

      page_range_type const& buffer_range() const
      { return buffer_range_; }

      template <typename StateInteger, typename BitInteger>
      bool is_page_qubit(::ket::qubit<StateInteger, BitInteger> const permutated_qubit) const
      {
        return static_cast<BitInteger>(permutated_qubit) >= num_local_qubits_ - num_page_qubits
          and static_cast<BitInteger>(permutated_qubit) < num_local_qubits_;
      }

      std::size_t num_local_qubits() const { return num_local_qubits_; }

      bool operator==(state const& other) const { return data_ == other.data_; }
      bool operator<(state const& other) const { return data_ < other.data_; }

      // Element access
      reference at(size_type const index)
      {
        return data_.at(
          (::ket::utility::begin(page_ranges_[get_page_id(index)]) - ::ket::utility::begin(data_))
          + get_index_in_page(index));
      }

      const_reference at(size_type const index) const
      {
        return data_.at(
          (::ket::utility::begin(page_ranges_[get_page_id(index)]) - ::ket::utility::begin(data_))
          + get_index_in_page(index));
      }

      reference operator[](size_type const index)
      {
        assert(index < (size_type{1u} << num_local_qubits_));
        return ::ket::utility::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      const_reference operator[](size_type const index) const
      {
        assert(index < (size_type{1u} << num_local_qubits_));
        return ::ket::utility::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      reference front() { return *::ket::utility::begin(page_ranges_[get_page_id(0u)]); }
      const_reference front() const { return *::ket::utility::begin(page_ranges_[get_page_id(0u)]); }

      reference back() { return *--::ket::utility::end(page_ranges_[get_page_id((1u << num_local_qubits_) - 1u)]); }
      const_reference back() const { return *--::ket::utility::end(page_ranges_[get_page_id((1u << num_local_qubits_) - 1u)]); }

      // Iterators
      iterator begin() noexcept { return iterator(*this, 0); }
      const_iterator begin() const noexcept { return const_iterator(*this, 0); }
      const_iterator cbegin() const noexcept { return const_iterator(*this, 0); }
      iterator end() noexcept
      { return iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      const_iterator end() const noexcept
      { return const_iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      const_iterator cend() const noexcept
      { return const_iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      reverse_iterator rbegin() noexcept { return reverse_iterator{this->end()}; }
      const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator{this->end()}; }
      const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator{this->cend()}; }
      reverse_iterator rend() noexcept { return reverse_iterator{this->begin()}; }
      const_reverse_iterator rend() const noexcept { return const_reverse_iterator{this->begin()}; }
      const_reverse_iterator crend() const noexcept { return const_reverse_iterator{this->cbegin()}; }

      // Capacity
      bool empty() const noexcept { return data_.empty(); }
      size_type size() const noexcept { return data_.size() - boost::size(buffer_range_); }
      size_type max_size() const noexcept { return data_.max_size() - boost::size(buffer_range_); }
      void reserve(size_type const new_capacity) { data_.reserve(new_capacity + boost::size(buffer_range_)); }
      size_type capacity() const noexcept { return data_.capacity() - boost::size(buffer_range_); }
      void shrink_to_fit() { data_.shrink_to_fit(); }

      // Modifiers
      void swap(state& other)
        noexcept(
          KET_is_nothrow_swappable<data_type>::value
          and KET_is_nothrow_swappable<std::size_t>::value
          and KET_is_nothrow_swappable< std::array<page_range_type, num_pages> >::value
          and KET_is_nothrow_swappable<page_range_type>::value )
      {
        using std::swap;
        swap(data_, other.data_);

        swap(num_local_qubits_, other.num_local_qubits_);
        swap(page_ranges_, other.page_ranges_);
        swap(buffer_range_, other.buffer_range_);
      }

     private:
      data_type generate_initial_data(
        std::initializer_list<value_type> initializer_list,
        allocator_type const& allocator) const
      {
        auto result = data_type{allocator};

        auto const state_size = initializer_list.size();
        auto const result_size = state_size + state_size / num_pages;

        assert(state_size % num_pages == 0);

        result.reserve(result_size);
        result.assign(initializer_list);
        result.resize(result_size);
        return result;
      }

      template <
        typename MpiPolicy, typename BitInteger, typename StateInteger,
        typename PermutationAllocator>
      data_type generate_initial_data(
        MpiPolicy const mpi_policy,
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment) const
      {
        auto result = data_type{};

        auto const state_size = ::ket::utility::integer_exp2<std::size_t>(num_local_qubits);
        auto const result_size = state_size + state_size / num_pages;

        assert(state_size % num_pages == 0);

        result.reserve(result_size);
        result.assign(state_size, value_type{0});

        using ::ket::mpi::permutate_bits;
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, result, permutate_bits(permutation, initial_integer));

        if (communicator.rank(environment) == rank_index.first)
          result[rank_index.second] = value_type{1};

        result.resize(result_size);
        return result;
      }

      std::array<page_range_type, num_pages>
      generate_initial_page_ranges(data_type& data) const
      {
        assert(data.size() % (num_pages + 1u) == 0u);
        auto const page_size = static_cast<size_type>(data.size() / (num_pages + 1u));

        auto result = std::array<page_range_type, num_pages>{};
        for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
        {
          result[page_id]
            = boost::make_iterator_range(
                ::ket::utility::begin(data) + page_id * page_size,
                ::ket::utility::begin(data) + (page_id + 1u) * page_size);
        }

        return result;
      }

      page_range_type generate_initial_buffer_range(data_type& data) const
      {
        assert(data.size() % (num_pages + 1u) == 0u);
        auto const page_size = static_cast<size_type>(data.size() / (num_pages + 1u));

        return boost::make_iterator_range(
          ::ket::utility::begin(data) + num_pages * page_size,
          ::ket::utility::begin(data) + (num_pages + 1u) * page_size);
      }

     public:
      size_type get_page_id(size_type const index) const
      {
        assert(index < (size_type{1u} << num_local_qubits_));

        auto const num_qubits_in_page = num_local_qubits_ - num_page_qubits;
        return (((num_pages - 1u) << num_qubits_in_page) bitand index) >> num_qubits_in_page;
      }

      size_type get_index_in_page(size_type const index) const
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));

        auto const num_qubits_in_page = num_local_qubits_ - num_page_qubits;
        return (compl ((num_pages - 1u) << num_qubits_in_page)) bitand index;
      }
    }; // class state<Complex, num_page_qubits, Allocator>

    namespace state_detail
    {
      template <
        typename Complex, int num_page_qubits, typename Allocator,
        typename StateInteger, typename BitInteger>
      void interpage_swap(
        ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
        ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
      {
        static_assert(num_page_qubits >= 2, "num_page_qubits should be at least 2 if using this function");
        assert(permutated_qubit1 != permutated_qubit2);

        auto const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits);
        auto const minmax_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto const lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first - static_cast<qubit_type>(num_nonpage_qubits))
            - StateInteger{1u};
        auto const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(
               minmax_qubits.second - static_cast<qubit_type>(num_nonpage_qubits + 1u))
             - StateInteger{1u})
            xor lower_bits_mask;
        auto const upper_bits_mask
          = compl (lower_bits_mask bitor middle_bits_mask);

        for (auto value_wo_qubits = StateInteger{0u};
             value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_page_qubits - 2));
             ++value_wo_qubits)
        {
          auto const base_page_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          auto const page_index1
            = base_page_index
              bitor (StateInteger{1u}
                     << (permutated_qubit1 - static_cast<qubit_type>(num_nonpage_qubits)));
          auto const page_index2
            = base_page_index
              bitor (StateInteger{1u}
                     << (permutated_qubit2 - static_cast<qubit_type>(num_nonpage_qubits)));

          local_state.swap_pages(page_index1, page_index2);
        }
      }

      template <
        typename ParallelPolicy, typename Complex, int num_page_qubits, typename Allocator,
        typename StateInteger, typename BitInteger>
      void swap_page_and_nonpage_qubits(
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
        ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
      {
        static_assert(num_page_qubits >= 1, "num_page_qubits should be at least 1 if using this function");

        auto const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits);
        auto const minmax_qubits = std::minmax(permutated_qubit1, permutated_qubit2);
        auto const nonpage_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.first) - StateInteger{1u};
        auto const nonpage_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(num_nonpage_qubits - 1u) - StateInteger{1u})
            xor nonpage_lower_bits_mask;
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto const page_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(minmax_qubits.second - static_cast<qubit_type>(num_nonpage_qubits))
            - StateInteger{1u};
        auto const page_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(num_page_qubits - 1u) - StateInteger{1u})
            xor page_lower_bits_mask;

        for (auto page_value_wo_qubits = StateInteger{0u};
             page_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_page_qubits - 1u));
             ++page_value_wo_qubits)
        {
          auto const page_index0
            = ((page_value_wo_qubits bitand page_upper_bits_mask) << 1u)
              bitor (page_value_wo_qubits bitand page_lower_bits_mask);
          auto const page_index1
            = (StateInteger{1u} << (minmax_qubits.second - static_cast<qubit_type>(num_nonpage_qubits)))
              bitor page_index0;

          for (auto nonpage_value_wo_qubits = StateInteger{0u};
               nonpage_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_nonpage_qubits - 1u));
               ++nonpage_value_wo_qubits)
          {
            auto const nonpage_index0
              = ((nonpage_value_wo_qubits bitand nonpage_upper_bits_mask) << 1u)
                bitor (nonpage_value_wo_qubits bitand nonpage_lower_bits_mask);
            auto const nonpage_index1
              = nonpage_index0 bitor (StateInteger{1u} << minmax_qubits.first);

            local_state.swap_values(
              std::make_pair(page_index0, nonpage_index1),
              std::make_pair(page_index1, nonpage_index0));
          }
        }
      }

      template <int num_page_qubits>
      struct swap_local_qubits
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger>
        static void call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
        {
          static_assert(num_page_qubits >= 2, "num_page_qubits should be at least 2 if using this function");
          if (local_state.is_page_qubit(permutated_qubit1))
          {
            if (local_state.is_page_qubit(permutated_qubit2))
              ::ket::mpi::state_detail::interpage_swap(
                local_state, permutated_qubit1, permutated_qubit2);
            else
              ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
          else if (local_state.is_page_qubit(permutated_qubit2))
            ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1);
          else
          {
            // In the process of "make_local_swap_qubit_swappable, it should not come to this clause
            typedef std::vector<Complex, Allocator> dummy_local_state_type;
            ::ket::mpi::utility::dispatch::swap_local_qubits<dummy_local_state_type>::call(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
        }
      }; // struct swap_local_qubits<num_page_qubits>

      template <>
      struct swap_local_qubits<1>
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger>
        static void call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 1, Allocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
        {
          if (local_state.is_page_qubit(permutated_qubit1))
            ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          else if (local_state.is_page_qubit(permutated_qubit2))
            ::ket::mpi::state_detail::swap_page_and_nonpage_qubits(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1);
          else
          {
            // In the process of "make_local_swap_qubit_swappable, it should not come to this clause
            typedef std::vector<Complex, Allocator> dummy_local_state_type;
            ::ket::mpi::utility::dispatch::swap_local_qubits<dummy_local_state_type>::call(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
        }
      }; // struct swap_local_qubits<1>

      template <int num_page_qubits>
      struct interchange_qubits
      {
        static_assert(
          num_page_qubits >= 1,
          "num_page_qubits should be at least 1 if using this function");

        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger>
        static void call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          std::vector<Complex, Allocator_>&,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using page_range_type
            = typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::page_range_type;
          using page_iterator
            = typename ::ket::utility::meta::iterator_of<page_range_type>::type;
          do_call(
            local_state, source_local_first_index, source_local_last_index,
            [target_rank, &communicator, &environment](
              page_iterator const first, page_iterator const last,
              page_iterator const buffer_first, page_iterator const buffer_last)
            {
              yampi::algorithm::swap(
                yampi::ignore_status(),
                yampi::make_buffer(first, last),
                yampi::make_buffer(buffer_first, buffer_last),
                target_rank, communicator, environment);
            });
        }

        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger,
          typename DerivedDatatype>
        static void call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          std::vector<Complex, Allocator_>&,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using page_range_type
            = typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::page_range_type;
          using page_iterator
            = typename ::ket::utility::meta::iterator_of<page_range_type>::type;
          do_call(
            local_state, source_local_first_index, source_local_last_index,
            [&datatype, target_rank, &communicator, &environment](
              page_iterator const first, page_iterator const last,
              page_iterator const buffer_first, page_iterator const buffer_last)
            {
              yampi::algorithm::swap(
                yampi::ignore_status(),
                yampi::make_buffer(first, last, datatype),
                yampi::make_buffer(buffer_first, buffer_last, datatype),
                target_rank, communicator, environment);
            });
        }

       private:
        template <
          typename Allocator, typename Complex, typename StateInteger, typename Function>
        static void do_call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          StateInteger const source_local_first_index, StateInteger const source_local_last_index,
          Function&& yampi_swap)
        {
          assert(source_local_last_index >= source_local_first_index);

          auto const page_front_id = local_state.get_page_id(source_local_first_index);
          auto const page_back_id = local_state.get_page_id(source_local_last_index - 1u);

          for (auto page_id = page_front_id; page_id <= page_back_id; ++page_id)
          {
            auto page_range = local_state.page_range(page_id);
            auto const page_size = boost::size(page_range);

            auto const page_first = ::ket::utility::begin(page_range);
            auto const page_last = ::ket::utility::end(page_range);
            auto const buffer_first = ::ket::utility::begin(local_state.buffer_range());

            auto const first_index
              = page_id == page_front_id
                ? static_cast<StateInteger>(
                    local_state.get_index_in_page(source_local_first_index))
                : StateInteger{0u};
            auto const last_index
              = page_id == page_back_id
                ? static_cast<StateInteger>(
                    local_state.get_index_in_page(source_local_last_index - 1u) + 1u)
                : static_cast<StateInteger>(page_size);

            auto const the_first = page_first + first_index;
            auto const the_last = page_first + last_index;
            auto const the_buffer_first = buffer_first + first_index;
            auto const the_buffer_last = buffer_first + last_index;

            std::copy(page_first, the_first, buffer_first);
            std::copy(the_last, page_last, the_buffer_last);

            yampi_swap(the_first, the_last, the_buffer_first, the_buffer_last);

            local_state.swap_buffer_and_page(page_id);
          }
        }
      }; // struct interchange_qubits<num_page_qubits>

      template <int num_page_qubits>
      struct for_each_local_range
      {
        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, num_page_qubits, Allocator>& call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          Function&& function)
        {
          // Gates should not be on page qubits
          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
            function(
              ::ket::utility::begin(local_state.page_range(page_id)),
              ::ket::utility::end(local_state.page_range(page_id)));
          return local_state;
        }

        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          Function&& function)
        {
          // Gates should not be on page qubits
          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
            function(
              ::ket::utility::begin(local_state.page_range(page_id)),
              ::ket::utility::end(local_state.page_range(page_id)));
          return local_state;
        }
      }; // struct for_each_local_range<num_page_qubits>

# ifdef KET_USE_DIAGONAL_LOOP
      template <int num_page_qubits>
      struct diagonal_loop
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator,
          typename Function0, typename Function1, typename... ControlQubits>
        static void call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::communicator const& communicator,
          yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          Function0&& function0, Function1&& function1,
          ControlQubits... control_qubits)
        {
          // Gates should not be on page qubits
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto local_permutated_control_qubits = std::array<qubit_type, 0u>{};

          auto const least_global_permutated_qubit
            = qubit_type{::ket::utility::integer_log2<BitInteger>(boost::size(local_state))};

          call_impl(
            parallel_policy, local_state, permutation, communicator.rank(environment),
            least_global_permutated_qubit, target_qubit,
            std::forward<Function0>(function0),
            std::forward<Function1>(function1),
            local_permutated_control_qubits, control_qubits...);
        }

       private:
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator,
          typename Function0, typename Function1,
          std::size_t num_local_control_qubits, typename... ControlQubits>
        static void call_impl(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank,
          ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          Function0&& function0, Function1&& function1,
          std::array<
            ::ket::qubit<StateInteger, BitInteger>,
            num_local_control_qubits> const& local_permutated_control_qubits,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ControlQubits... control_qubits)
        {
          auto const permutated_control_qubit = permutation[control_qubit.qubit()];

          if (permutated_control_qubit < least_global_permutated_qubit)
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_local_permutated_control_qubits
              = std::array<qubit_type, num_local_control_qubits + 1u>{};
            std::copy(
              ::ket::utility::begin(local_permutated_control_qubits),
              ::ket::utility::end(local_permutated_control_qubits),
              ::ket::utility::begin(new_local_permutated_control_qubits));
            new_local_permutated_control_qubits.back() = permutated_control_qubit;

            call_impl(
              parallel_policy, local_state, permutation, rank,
              least_global_permutated_qubit, target_qubit,
              std::forward<Function0>(function0),
              std::forward<Function1>(function1),
              new_local_permutated_control_qubits, control_qubits...);
          }
          else
          {
            static constexpr auto zero_state_integer = StateInteger{0u};
            static constexpr auto one_state_integer = StateInteger{1u};

            auto const mask
              = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) != zero_state_integer)
              call_impl(
                parallel_policy, local_state, permutation, rank,
                least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0),
                std::forward<Function1>(function1),
                local_permutated_control_qubits, control_qubits...);
          }
        }

        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator,
          typename Function0, typename Function1,
          std::size_t num_local_control_qubits>
        static void call_impl(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank,
          ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          Function0&& function0, Function1&& function1,
          std::array<
            ::ket::qubit<StateInteger, BitInteger>,
            num_local_control_qubits> const& local_permutated_control_qubits)
        {
          auto const permutated_target_qubit = permutation[target_qubit];

          static constexpr auto one_state_integer = StateInteger{1u};

          auto const last_integer
            = ((one_state_integer << least_global_permutated_qubit) >> num_page_qubits) >> num_local_control_qubits;

          if (permutated_target_qubit < least_global_permutated_qubit)
          {
            auto const mask = one_state_integer << permutated_target_qubit;

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
            for_each(
              parallel_policy, local_state, last_integer, local_permutated_control_qubits,
              [&function0, &function1, mask](auto const iter, StateInteger const state_integer)
              {
                static constexpr auto zero_state_integer = StateInteger{0u};

                if ((state_integer bitand mask) == zero_state_integer)
                  function0(iter, state_integer);
                else
                  function1(iter, state_integer);
              });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
            for_each(
              parallel_policy, local_state, last_integer, local_permutated_control_qubits,
              make_call_function_if_local(std::forward<Function0>(function0), std::forward<Function1>(function1), mask));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
          }
          else
          {
            auto const mask
              = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

            static constexpr auto zero_state_integer = StateInteger{0u};

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
              for_each(
                parallel_policy, local_state, last_integer, local_permutated_control_qubits,
                std::forward<Function0>(function0));
            else
              for_each(
                parallel_policy, local_state, last_integer, local_permutated_control_qubits,
                std::forward<Function1>(function1));
          }
        }

#   ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename Function0, typename Function1, typename StateInteger>
        struct call_function_if_local
        {
          Function0 function0_;
          Function1 function1_;
          StateInteger mask_;

          template <typename Iterator>
          void operator()(Iterator const iter, StateInteger const state_integer)
          {
            static constexpr auto zero_state_integer = StateInteger{0u};

            if ((state_integer bitand mask_) == zero_state_integer)
              function0_(iter, state_integer);
            else
              function1_(iter, state_integer);
          }
        }; // struct call_function_if_local<Function0, Function1, StateInteger>

        template <typename Function0, typename Function1, typename StateInteger>
        static call_function_if_local<Function0, Function1, StateInteger>
        make_call_function_if_local(Function0&& function0, Function1&& function1, StateInteger const mask)
        { return call_function_if_local<Function0, Function1, StateInteger>{std::forward<Function0>(function0), std::forward<Function1>(function1), mask}; }
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger,
          std::size_t num_local_control_qubits, typename Function>
        static void for_each(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          StateInteger const last_integer,
          std::array<
            ::ket::qubit<StateInteger, BitInteger>,
            num_local_control_qubits> const& local_permutated_control_qubits,
          Function&& function)
        {
          auto sorted_local_permutated_control_qubits = local_permutated_control_qubits;
          std::sort(
            ::ket::utility::begin(sorted_local_permutated_control_qubits),
            ::ket::utility::end(sorted_local_permutated_control_qubits));

          for_each_impl(
            parallel_policy, local_state, last_integer, sorted_local_permutated_control_qubits,
            std::forward<Function>(function));
        }

        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger,
          std::size_t num_local_control_qubits, typename Function>
        static void for_each_impl(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          StateInteger const last_integer,
          std::array<
            ::ket::qubit<StateInteger, BitInteger>,
            num_local_control_qubits> const& sorted_local_permutated_control_qubits,
          Function&& function)
        {
          static constexpr auto zero_state_integer = StateInteger{0u};

          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          // 000101000100
          auto const mask
            = std::accumulate(
                ::ket::utility::begin(sorted_local_permutated_control_qubits),
                ::ket::utility::end(sorted_local_permutated_control_qubits),
                zero_state_integer,
                [](StateInteger const& partial_mask, qubit_type const& control_qubit)
                {
                  static constexpr auto one_state_integer = StateInteger{1u};
                  return partial_mask bitor (one_state_integer << control_qubit);
                });

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            auto const first = ::ket::utility::begin(local_state.page_range(page_id));
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy, last_integer,
              [&function, &sorted_local_permutated_control_qubits, mask, first](StateInteger state_integer, int const)
              {
                static constexpr auto one_state_integer = StateInteger{1u};

                // xxx0x0xxx0xx
                for (qubit_type const& qubit: sorted_local_permutated_control_qubits)
                {
                  auto const lower_mask = (one_state_integer << qubit) - one_state_integer;
                  auto const upper_mask = compl lower_mask;
                  state_integer
                    = (state_integer bitand lower_mask)
                      bitor ((state_integer bitand upper_mask) << 1u);
                }

                // xxx1x1xxx1xx
                state_integer |= mask;

                function(first + state_integer, state_integer);
              });
          }
        }
      }; // struct diagonal_loop<num_page_qubits>
# endif // KET_USE_DIAGONAL_LOOP

      template <int num_page_qubits>
      struct transform_inclusive_scan
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
#   ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <
          typename PartialSums, typename ParallelPolicy,
          typename LocalState, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        class process_in_execute
        {
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          ParallelPolicy parallel_policy_;
          LocalState& local_state_;
          ForwardIterator d_first_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;

         public:
          process_in_execute(
            unsigned int num_threads, PartialSums& partial_sums,
            ParallelPolicy parallel_policy,
            LocalState& local_state, ForwardIterator d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation)
            : num_threads_{num_threads},
              partial_sums_{partial_sums},
              parallel_policy_{parallel_policy},
              local_state_{local_state},
              d_first_{d_first},
              binary_operation_{binary_operation},
              unary_operation_{unary_operation}
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            auto d_page_first = d_first_;

            for (auto page_id = std::size_t{0u}; page_id < LocalState::num_pages; ++page_id)
            {
              auto const first = ::ket::utility::begin(local_state_.page_range(page_id));
              auto d_iter = d_page_first;
              auto is_called = false;

              using difference_type
                = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                [page_id, first, &d_iter, &is_called, this](
                  difference_type const n, int const thread_index)
                {
                  if (not is_called)
                  {
                    std::advance(d_iter, n);
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->unary_operation_(first[n]);
                    is_called = true;
                  }
                  else
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->binary_operation_(
                          this->partial_sums_[this->num_threads_ * page_id + thread_index],
                          this->unary_operation_(first[n]));

                  *d_iter++ = this->partial_sums_[this->num_threads_ * page_id + thread_index];
                });

              std::advance(d_page_first, boost::size(local_state_.page_range(page_id)));
            }

            post_process(
              parallel_policy_, local_state_, d_first_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        }; // struct process_in_execute<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation>

        template <
          typename PartialSums, typename ParallelPolicy,
          typename LocalState, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static process_in_execute<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation>
        make_process_in_execute(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy,
          LocalState& local_state, ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          using result_type
            = process_in_execute<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation>;
          return result_type{
            num_threads, partial_sums, parallel_policy, local_state, d_first, binary_operation, unary_operation};
        }

        template <
          typename PartialSums, typename ParallelPolicy,
          typename LocalState, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        class process_in_execute_with_initial_value
        {
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          ParallelPolicy parallel_policy_;
          LocalState& local_state_;
          ForwardIterator d_first_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;
          Complex initial_value_;

         public:
          process_in_execute_with_initial_value(
            unsigned int num_threads, PartialSums& partial_sums,
            ParallelPolicy parallel_policy,
            LocalState& local_state, ForwardIterator d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex initial_value)
            : num_threads_{num_threads},
              partial_sums_{partial_sums},
              parallel_policy_{parallel_policy},
              local_state_{local_state},
              d_first_{d_first},
              binary_operation_{binary_operation},
              unary_operation_{unary_operation},
              initial_value_{initial_value}
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            auto d_page_first = d_first_;

            for (auto page_id = std::size_t{0u}; page_id < LocalState::num_pages; ++page_id)
            {
              auto const first = ::ket::utility::begin(local_state_.page_range(page_id));
              auto d_iter = d_page_first;
              auto is_called = false;

              using difference_type
                = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                [page_id, first, &d_iter, &is_called, this](
                  difference_type const n, int const thread_index)
                {
                  if (not is_called)
                  {
                    std::advance(d_iter, n);
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = thread_index == 0
                        ? this->binary_operation_(this->initial_value_, this->unary_operation_(first[n]))
                        : this->unary_operation_(first[n]);
                    is_called = true;
                  }
                  else
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->binary_operation_(
                          this->partial_sums_[this->num_threads_ * page_id + thread_index],
                          this->unary_operation_(first[n]));

                  *d_iter++ = this->partial_sums_[this->num_threads_ * page_id + thread_index];
                });

              std::advance(d_page_first, boost::size(local_state_.page_range(page_id)));
            }

            post_process(
              parallel_policy_, local_state_, d_first_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        }; // struct process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation, Complex>

        template <
          typename PartialSums, typename ParallelPolicy,
          typename LocalState, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static process_in_execute_with_initial_value<
          PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation, Complex>
        make_process_in_execute_with_initial_value(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy,
          LocalState& local_state, ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex initial_value)
        {
          using result_type
            = process_in_execute_with_initial_value<
                PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation, Complex>;
          return result_type{
            num_threads, partial_sums, parallel_policy,
            local_state, d_first, binary_operation, unary_operation, initial_value};
        }
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        {
          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Complex>(num_threads * num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              auto d_page_first = d_first;

              for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
              {
                auto const first = ::ket::utility::begin(local_state.page_range(page_id));
                auto d_iter = d_page_first;
                auto is_called = false;

                using difference_type
                  = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  [page_id, first, &d_iter, &is_called, num_threads, &partial_sums,
                   binary_operation, unary_operation](
                    difference_type const n, int const thread_index)
                  {
                    if (not is_called)
                    {
                      std::advance(d_iter, n);
                      partial_sums[num_threads * page_id + thread_index]
                        = unary_operation(first[n]);
                      is_called = true;
                    }
                    else
                      partial_sums[num_threads * page_id + thread_index]
                        = binary_operation(
                            partial_sums[num_threads * page_id + thread_index],
                            unary_operation(first[n]));

                    *d_iter++ = partial_sums[num_threads * page_id + thread_index];
                  });

                std::advance(d_page_first, boost::size(local_state.page_range(page_id)));
              }

              post_process(
                parallel_policy, local_state, d_first, binary_operation,
                partial_sums, thread_index, executor);
            });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            make_process_in_execute(
              num_threads, partial_sums, parallel_policy, local_state, d_first,
              binary_operation, unary_operation));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return partial_sums.back();
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        {
          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Complex>(num_threads * num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              auto d_page_first = d_first;

              for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
              {
                auto const first = ::ket::utility::begin(local_state.page_range(page_id));
                auto d_iter = d_page_first;
                auto is_called = false;

                using difference_type
                  = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  [page_id, first, &d_iter, &is_called, num_threads, &partial_sums,
                   binary_operation, unary_operation, initial_value](
                    difference_type const n, int const thread_index)
                  {
                    if (not is_called)
                    {
                      std::advance(d_iter, n);
                      partial_sums[num_threads * page_id + thread_index]
                        = thread_index == 0
                          ? binary_operation(initial_value, unary_operation(first[n]))
                          : unary_operation(first[n]);
                      is_called = true;
                    }
                    else
                      partial_sums[num_threads * page_id + thread_index]
                        = binary_operation(
                            partial_sums[num_threads * page_id + thread_index],
                            unary_operation(first[n]));

                    *d_iter++ = partial_sums[num_threads * page_id + thread_index];
                  });

                std::advance(d_page_first, boost::size(local_state.page_range(page_id)));
              }

              post_process(
                parallel_policy, local_state, d_first, binary_operation,
                partial_sums, thread_index, executor);
            });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            make_process_in_execute_with_initial_value(
              num_threads, partial_sums, parallel_policy, local_state, d_first,
              binary_operation, unary_operation, initial_value));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return partial_sums.back();
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename Executor>
        static void post_process(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          ForwardIterator d_first, BinaryOperation binary_operation,
          std::vector<Complex>& partial_sums, int const thread_index,
          Executor& executor)
        {
          ::ket::utility::barrier(parallel_policy, executor);

          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              std::partial_sum(
                ::ket::utility::begin(partial_sums), ::ket::utility::end(partial_sums),
                ::ket::utility::begin(partial_sums), binary_operation);
            });

          auto d_page_first = d_first;

          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          using local_state_type = ::ket::mpi::state<Complex, num_page_qubits, Allocator>;
          static constexpr auto num_pages = local_state_type::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            auto d_iter = d_page_first;
            auto is_called = false;

            using page_range_type = typename local_state_type::page_range_type;
            using page_iterator = typename boost::range_iterator<page_range_type>::type;
            using difference_type = typename std::iterator_traits<page_iterator>::difference_type;
            ::ket::utility::loop_n_in_execute(
              parallel_policy,
              boost::size(local_state.page_range(page_id)), thread_index,
              [page_id, &d_iter, &is_called,
               num_threads, &partial_sums, binary_operation](
                difference_type const n, int const thread_index)
              {
                if (thread_index == 0u and page_id == 0u)
                  return;

                if (not is_called)
                {
                  std::advance(d_iter, n);
                  is_called = true;
                }

                *d_iter
                  = binary_operation(
                      partial_sums[num_threads * page_id + thread_index - 1u],
                      *d_iter);
                ++d_iter;
              });

            std::advance(d_page_first, boost::size(local_state.page_range(page_id)));
          }
        }
# else // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
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
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
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
        static Complex impl(
          std::forward_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        {
          auto prev_d_first = d_first;
          d_first
            = ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy,
                local_state.page_range(0u), d_first, binary_operation, unary_operation);
          std::advance(prev_d_first, boost::size(local_state.page_range(0u)) - 1);
          auto partial_sum = *prev_d_first;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{1u}; page_id < num_pages; ++page_id)
          {
            prev_d_first = d_first;
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, boost::size(local_state.page_range(page_id)) - 1);
            partial_sum = *prev_d_first;
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex impl(
          std::forward_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        {
          auto partial_sum = initial_value;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            auto prev_d_first = d_first;
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, boost::size(local_state.page_range(page_id)) - 1);
            partial_sum = *prev_d_first;
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BidirectionalIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex impl(
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          BidirectionalIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        {
          d_first
            = ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy,
                local_state.page_range(0u), d_first, binary_operation, unary_operation);
          auto partial_sum = *std::prev(d_first);

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{1u}; page_id < num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(d_first);
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BidirectionalIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex impl(
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          BidirectionalIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        {
          auto partial_sum = initial_value;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(d_first);
          }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      }; // struct transform_inclusive_scan<num_page_qubits>

      template <int num_page_qubits>
      struct transform_inclusive_scan_self
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
#   ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation>
        class process_in_execute
        {
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          ParallelPolicy parallel_policy_;
          LocalState& local_state_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;

         public:
          process_in_execute(
            unsigned int num_threads, PartialSums& partial_sums,
            ParallelPolicy parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation)
            : num_threads_{num_threads},
              partial_sums_{partial_sums},
              parallel_policy_{parallel_policy},
              local_state_{local_state},
              binary_operation_{binary_operation},
              unary_operation_{unary_operation}
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            for (auto page_id = std::size_t{0u}; page_id < LocalState::num_pages; ++page_id)
            {
              auto const first = ::ket::utility::begin(local_state_.page_range(page_id));
              auto is_called = false;

              using difference_type
                = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                [page_id, first, &is_called, this](difference_type const n, int const thread_index)
                {
                  if (not is_called)
                  {
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->unary_operation_(first[n]);
                    is_called = true;
                  }
                  else
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->binary_operation_(
                          this->partial_sums_[this->num_threads_ * page_id + thread_index],
                          this->unary_operation_(first[n]));

                  first[n] = this->partial_sums_[this->num_threads_ * page_id + thread_index];
                });
            }

            post_process(
              parallel_policy_, local_state_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        }; // struct process_in_execute<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation>

        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation>
        static process_in_execute<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation>
        make_process_in_execute(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy, LocalState& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          using result_type
            = process_in_execute<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation>;
          return result_type{
            num_threads, partial_sums, parallel_policy, local_state, binary_operation, unary_operation};
        }

        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        class process_in_execute_with_initial_value
        {
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          ParallelPolicy parallel_policy_;
          LocalState& local_state_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;
          Complex initial_value_;

         public:
          process_in_execute_with_initial_value(
            unsigned int num_threads, PartialSums& partial_sums,
            ParallelPolicy parallel_policy, LocalState& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex initial_value)
            : num_threads_{num_threads},
              partial_sums_{partial_sums},
              parallel_policy_{parallel_policy},
              local_state_{local_state},
              binary_operation_{binary_operation},
              unary_operation_{unary_operation},
              initial_value_{initial_value}
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            for (auto page_id = std::size_t{0u}; page_id < LocalState::num_pages; ++page_id)
            {
              auto const first = ::ket::utility::begin(local_state_.page_range(page_id));
              auto is_called = false;

              using difference_type
                = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                [page_id, first, &is_called, this](difference_type const n, int const thread_index)
                {
                  if (not is_called)
                  {
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = page_id == 0 && thread_index == 0
                        ? this->binary_operation_(this->initial_value_, this->unary_operation_(first[n]))
                        : this->unary_operation_(first[n]);
                    is_called = true;
                  }
                  else
                    this->partial_sums_[this->num_threads_ * page_id + thread_index]
                      = this->binary_operation_(
                          this->partial_sums_[this->num_threads_ * page_id + thread_index],
                          this->unary_operation_(first[n]));

                  first[n] = this->partial_sums_[this->num_threads_ * page_id + thread_index];
                });
            }

            post_process(
              parallel_policy_, local_state_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        }; // struct process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation, Complex>

        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static process_in_execute_with_initial_value<
          PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation, Complex>
        make_process_in_execute_with_initial_value(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy, LocalState& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Complex initial_value)
        {
          using result_type
            = process_in_execute_with_initial_value<
                PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation, Complex>;
          return result_type{
            num_threads, partial_sums, parallel_policy,
            local_state, binary_operation, unary_operation, initial_value};
        }
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        {
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          static constexpr auto num_pages = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          auto partial_sums = std::vector<Complex>(num_threads * num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
              {
                auto const first = ::ket::utility::begin(local_state.page_range(page_id));
                bool is_called = false;

                using difference_type
                  = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  [page_id, first, &is_called, num_threads, &partial_sums,
                   binary_operation, unary_operation](
                    difference_type const n, int const thread_index)
                  {
                    if (not is_called)
                    {
                      partial_sums[num_threads * page_id + thread_index]
                        = unary_operation(first[n]);
                      is_called = true;
                    }
                    else
                      partial_sums[num_threads * page_id + thread_index]
                        = binary_operation(
                            partial_sums[num_threads * page_id + thread_index],
                            unary_operation(first[n]));

                    first[n] = partial_sums[num_threads * page_id + thread_index];
                  });
              }

              post_process(
                parallel_policy, local_state, binary_operation,
                partial_sums, thread_index, executor);
            });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            make_process_in_execute(
              num_threads, partial_sums, parallel_policy, local_state,
              binary_operation, unary_operation));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return partial_sums.back();
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        {
          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          static constexpr auto num_pages = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          auto partial_sums = std::vector<Complex>(num_threads * num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
              {
                auto const first = ::ket::utility::begin(local_state.page_range(page_id));
                auto is_called = false;

                using difference_type
                  = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  [page_id, first, &is_called, num_threads, &partial_sums,
                   binary_operation, unary_operation, initial_value](
                    difference_type const n, int const thread_index)
                  {
                    if (not is_called)
                    {
                      partial_sums[num_threads * page_id + thread_index]
                        = page_id == 0 && thread_index == 0
                          ? binary_operation(initial_value, unary_operation(first[n]))
                          : unary_operation(first[n]);
                      is_called = true;
                    }
                    else
                      partial_sums[num_threads * page_id + thread_index]
                        = binary_operation(
                            partial_sums[num_threads * page_id + thread_index],
                            unary_operation(first[n]));

                    first[n] = partial_sums[num_threads * page_id + thread_index];
                  });
              }

              post_process(
                parallel_policy, local_state, binary_operation,
                partial_sums, thread_index, executor);
            });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            make_process_in_execute_with_initial_value(
              num_threads, partial_sums, parallel_policy, local_state,
              binary_operation, unary_operation, initial_value));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return partial_sums.back();
        }

       private:
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename BinaryOperation,
          typename Executor>
        static void post_process(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation,
          std::vector<Complex>& partial_sums, int const thread_index,
          Executor& executor)
        {
          ::ket::utility::barrier(parallel_policy, executor);

          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              std::partial_sum(
                ::ket::utility::begin(partial_sums), ::ket::utility::end(partial_sums),
                ::ket::utility::begin(partial_sums), binary_operation);
            });

          auto const num_threads = static_cast<unsigned int>(::ket::utility::num_threads(parallel_policy));
          using local_state_type = ::ket::mpi::state<Complex, num_page_qubits, Allocator>;
          static constexpr auto num_pages = local_state_type::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            auto const first = ::ket::utility::begin(local_state.page_range(page_id));

            using difference_type
              = typename std::iterator_traits<typename std::remove_cv<decltype(first)>::type>::difference_type;
            ::ket::utility::loop_n_in_execute(
              parallel_policy,
              boost::size(local_state.page_range(page_id)), thread_index,
              [page_id, first, num_threads, &partial_sums, binary_operation](
                difference_type const n, int const thread_index)
              {
                if (thread_index == 0u and page_id == 0u)
                  return;

                first[n]
                  = binary_operation(
                      partial_sums[num_threads * page_id + thread_index - 1u],
                      first[n]);
              });
          }
        }
# else // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const&)
        {
          ::ket::utility::ranges::transform_inclusive_scan(
            parallel_policy,
            local_state.page_range(0u),
            ::ket::utility::begin(local_state.page_range(0u)),
            binary_operation, unary_operation);
          auto partial_sum = *std::prev(::ket::utility::end(local_state.page_range(0u)));

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{1u}; page_id < num_pages; ++page_id)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state.page_range(page_id),
              ::ket::utility::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(::ket::utility::end(local_state.page_range(page_id)));
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const&)
        {
          auto partial_sum = initial_value;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits, Allocator>::num_pages;
          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state.page_range(page_id),
              ::ket::utility::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *std::prev(::ket::utility::end(local_state.page_range(page_id)));
          }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      }; // struct transform_inclusive_scan_self<num_page_qubits>

      // Usually num_page_qubits is small, so linear search for page is probably not bad.
      template <int num_page_qubits>
      struct upper_bound
      {
        template <typename Complex, typename Allocator, typename Compare>
        static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          Complex const& value, Compare compare, yampi::environment const&)
        {
          using local_state_type = ::ket::mpi::state<Complex, num_page_qubits, Allocator>;
          static constexpr auto num_pages = local_state_type::num_pages;
          using difference_type = typename local_state_type::difference_type;

          for (auto page_id = std::size_t{0u}; page_id < num_pages; ++page_id)
          {
            if (not compare(value, *std::prev(::ket::utility::end(local_state.page_range(page_id)))))
              continue;

            auto index_in_page
              = std::upper_bound(
                  ::ket::utility::begin(local_state.page_range(page_id)),
                  ::ket::utility::end(local_state.page_range(page_id)),
                  value, compare)
                - ::ket::utility::begin(local_state.page_range(page_id));

            auto const num_qubits_in_page
              = local_state.num_local_qubits() - num_page_qubits;
            return static_cast<difference_type>((page_id << num_qubits_in_page) bitor index_in_page);
          }

          return static_cast<difference_type>(local_state.size());
        }
      }; // struct upper_bound<num_page_qubits>
    } // namespace state_detail


    template <typename Complex, typename Allocator>
    class state<Complex, 0, Allocator>
    {
     public:
      using value_type = Complex;
      using allocator_type = typename Allocator::template rebind<value_type>::other;

      static constexpr auto num_pages = std::size_t{1u};

     private:
      using data_type = std::vector<value_type, allocator_type>;
      data_type data_;

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

      state() noexcept(noexcept(allocator_type{})) : data_{allocator_type{}} { }
      explicit state(allocator_type const& allocator) noexcept : data_{allocator} { }

      ~state() noexcept = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
      state(state&&) = default;
      state& operator=(state&&) = default;

      state(state const& other, allocator_type const& allocator)
        : data_{other.data_, allocator}
      { }

      state(state&& other, allocator_type const& allocator)
        : data_{std::move(other.data_), allocator}
      { }

      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_{initializer_list, allocator}
      { }

      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_{generate_initial_data(
            ::ket::mpi::utility::policy::make_general_mpi(),
            num_local_qubits, initial_integer, permutation, communicator, environment)}
      { }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_{generate_initial_data(
            mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment)}
      { }

      bool operator==(state const& other) const { return data_ == other.data_; }
      bool operator<(state const& other) const { return data_ < other.data_; }

      // Element access
      reference at(size_type const position) { return data_.at(position); }
      const_reference at(size_type const position) const { return data_.at(position); }
      reference operator[](size_type const position) { return data_[position]; }
      const_reference operator[](size_type const position) const { return data_[position]; }
      reference front() { return data_.front(); }
      const_reference front() const { return data_.front(); }
      reference back() { return data_.back(); }
      const_reference back() const { return data_.back(); }

      // Iterators
      iterator begin() noexcept { return data_.begin(); }
      const_iterator begin() const noexcept { return data_.begin(); }
      const_iterator cbegin() const noexcept { return data_.cbegin(); }
      iterator end() noexcept { return data_.end(); }
      const_iterator end() const noexcept { return data_.end(); }
      const_iterator cend() const noexcept { return data_.cend(); }
      reverse_iterator rbegin() noexcept { return data_.rbegin(); }
      const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
      const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
      reverse_iterator rend() noexcept { return data_.rend(); }
      const_reverse_iterator rend() const noexcept { return data_.rend(); }
      const_reverse_iterator crend() const noexcept { return data_.crend(); }

      // Capacity
      bool empty() const noexcept { return data_.empty(); }
      size_type size() const noexcept { return data_.size(); }
      size_type max_size() const noexcept { return data_.max_size(); } 
      void reserve(size_type const new_capacity) { data_.reserve(new_capacity); }
      size_type capacity() const noexcept { return data_.capacity(); }
      void shrink_to_fit() { data_.shrink_to_fit(); }

      // Modifiers
      void swap(state& other)
        noexcept(KET_is_nothrow_swappable<data_type>::value)
      {
        using std::swap;
        swap(data_, other.data_);
      }

      data_type& data() { return data_; }
      data_type const& data() const { return data_; }

     private:
      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      data_type generate_initial_data(
        MpiPolicy const mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment) const
      {
        auto result = data_type(::ket::utility::integer_exp2<std::size_t>(num_local_qubits), value_type{0});

        using ::ket::mpi::permutate_bits;
        auto const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, result, permutate_bits(permutation, initial_integer));

        if (communicator.rank(environment) == rank_index.first)
          result[rank_index.second] = value_type{1};

        return result;
      }
    }; // class state<Complex, 0, Allocator>

    template <typename Complex, int num_page_qubits, typename Allocator>
    inline bool operator!=(
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& rhs)
    { return not(lhs == rhs); }

    template <typename Complex, int num_page_qubits, typename Allocator>
    inline bool operator>(
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& rhs)
    { return rhs < lhs; }

    template <typename Complex, int num_page_qubits, typename Allocator>
    inline bool operator<=(
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& rhs)
    { return not(lhs > rhs); }

    template <typename Complex, int num_page_qubits, typename Allocator>
    inline bool operator>=(
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& lhs,
      ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& rhs)
    { return not(lhs < rhs); }

    template <typename Complex,  int num_page_qubits, typename Allocator>
    inline void swap(
      ::ket::mpi::state<Complex, num_page_qubits, Allocator>& lhs,
      ::ket::mpi::state<Complex, num_page_qubits, Allocator>& rhs)
      noexcept(
        KET_is_nothrow_swappable<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>>::value)
    { lhs.swap(rhs); }

    namespace state_detail
    {
      template <>
      struct swap_local_qubits<0>
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger>
        static void call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
        {
          ::ket::mpi::utility::detail::swap_local_qubits(
            parallel_policy, local_state.data(), permutated_qubit1, permutated_qubit2);
        }
      }; // struct swap_local_qubits<0>

      template <>
      struct interchange_qubits<0>
      {
        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger>
        static void call(
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          std::vector<Complex, Allocator_>& buffer,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          ::ket::mpi::utility::detail::interchange_qubits(
            local_state.data(), buffer, source_local_first_index, source_local_last_index,
            target_rank, communicator, environment);
        }

        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger,
          typename DerivedDatatype>
        static void call(
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          std::vector<Complex, Allocator_>& buffer,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          ::ket::mpi::utility::detail::interchange_qubits(
            local_state.data(), buffer, source_local_first_index, source_local_last_index,
            datatype, target_rank, communicator, environment);
        }
      }; // struct interchange_qubits<0>

      template <>
      struct for_each_local_range<0>
      {
        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, 0, Allocator>& call(
          ::ket::mpi::state<Complex, 0, Allocator>& local_state, Function&& function)
        {
          using dummy_local_state_type = std::vector<Complex, Allocator>;
          using for_each_local_range_type
            = ::ket::mpi::utility::dispatch::for_each_local_range<
                ::ket::mpi::utility::policy::general_mpi, dummy_local_state_type>;
          for_each_local_range_type::call(
            local_state.data(), std::forward<Function>(function));
          return local_state;
        }

        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, 0, Allocator> const& call(
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state, Function&& function)
        {
          using dummy_local_state_type = std::vector<Complex, Allocator>;
          using for_each_local_range_type
            = ::ket::mpi::utility::dispatch::for_each_local_range<
                ::ket::mpi::utility::policy::general_mpi, dummy_local_state_type>;
          for_each_local_range_type::call(
            local_state.data(), std::forward<Function>(function));
          return local_state;
        }
      }; // struct for_each_local_range<0>

# ifdef KET_USE_DIAGONAL_LOOP
      template <>
      struct diagonal_loop<0>
      {
        template <
          typename ParallelPolicy, typename Complex, typename Allocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator,
          typename Function0, typename Function1, typename... ControlQubits>
        static void call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::communicator const& communicator,
          yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          Function0&& function0, Function1&& function1,
          ControlQubits... control_qubits)
        {
          using dummy_local_state_type = std::vector<Complex, Allocator>;
          using diagonal_loop_type
            = ::ket::mpi::utility::dispatch::diagonal_loop<
                ::ket::mpi::utility::policy::general_mpi, dummy_local_state_type>;
          diagonal_loop_type::call(
            parallel_policy, local_state.data(), permutation, communicator, environment,
            target_qubit,
            std::forward<Function0>(function0), std::forward<Function1>(function1),
            control_qubits...);
        }
      }; // struct diagonal_loop<0>
# endif // KET_USE_DIAGONAL_LOOP

      template <>
      struct transform_inclusive_scan<0>
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
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
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::transform_inclusive_scan(
            parallel_policy,
            local_state.data(), d_first, binary_operation, unary_operation,
            initial_value, environment);
        }
      }; // struct transform_inclusive_scan<0>

      template <>
      struct transform_inclusive_scan_self<0>
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          yampi::environment const& environment)
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
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex const initial_value, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy,
            local_state.data(), binary_operation, unary_operation,
            initial_value, environment);
        }
      }; // struct transform_inclusive_scan_self<0>

      template <>
      struct upper_bound<0>
      {
        template <typename Complex, typename Allocator, typename Compare>
        static typename ::ket::mpi::state<Complex, 0, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          Complex const& value, Compare compare, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::upper_bound(
            local_state.data(), value, compare, environment);
        }
      }; // struct upper_bound<0>
    } // namespace state_detail

    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct swap_local_qubits;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct swap_local_qubits< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger>
          static void call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            ::ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
            ::ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
          {
            ::ket::mpi::state_detail::swap_local_qubits<num_page_qubits>::call(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
        }; // struct swap_local_qubits< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >

        template <typename LocalState_>
        struct interchange_qubits;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct interchange_qubits< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <typename Allocator_, typename StateInteger>
          static void call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            std::vector<Complex, Allocator_>& buffer,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            ::ket::mpi::state_detail::interchange_qubits<num_page_qubits>::call(
              local_state, buffer, source_local_first_index, source_local_last_index,
              target_rank, communicator, environment);
          }

          template <typename Allocator_, typename StateInteger, typename DerivedDatatype>
          static void call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            std::vector<Complex, Allocator_>& buffer,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            ::ket::mpi::state_detail::interchange_qubits<num_page_qubits>::call(
              local_state, buffer, source_local_first_index, source_local_last_index,
              datatype, target_rank, communicator, environment);
          }
        }; // struct interchange_qubits< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >

        template <typename MpiPolicy, typename LocalState_>
        struct for_each_local_range;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct for_each_local_range<
          ::ket::mpi::utility::policy::general_mpi,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <typename Function>
          static ::ket::mpi::state<Complex, num_page_qubits, Allocator>& call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state, Function&& function)
          {
            using for_each_local_range_type
              = ::ket::mpi::state_detail::for_each_local_range<num_page_qubits>;
            return for_each_local_range_type::call(
              local_state, std::forward<Function>(function));
          }

          template <typename Function>
          static ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state, Function&& function)
          {
            using for_each_local_range_type
              = ::ket::mpi::state_detail::for_each_local_range<num_page_qubits>;
            return for_each_local_range_type::call(
              local_state, std::forward<Function>(function));
          }
        }; // struct for_each_local_range< ::ket::mpi::utility::policy::general_mpi, ::ket::mpi::state<Complex, num_page_qubits, Allocator> >

# ifdef KET_USE_DIAGONAL_LOOP
        template <typename MpiPolicy, typename LocalState_>
        struct diagonal_loop;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct diagonal_loop<
          ::ket::mpi::utility::policy::general_mpi,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename PermutationAllocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits)
          {
            using diagonal_loop_type
              = ::ket::mpi::state_detail::diagonal_loop<num_page_qubits>;
            diagonal_loop_type::call(
              parallel_policy, local_state, permutation, communicator, environment,
              target_qubit,
              std::forward<Function0>(function0), std::forward<Function1>(function1),
              control_qubits...);
          }
        }; // struct diagonal_loop< ::ket::mpi::utility::policy::general_mpi, ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
# endif // KET_USE_DIAGONAL_LOOP

        template <typename LocalState_>
        struct transform_inclusive_scan;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct transform_inclusive_scan<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const& environment)
          {
            using transform_inclusive_scan_type
              = ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation,
              environment);
          }

          template <
            typename ParallelPolicy, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex const initial_value, yampi::environment const& environment)
          {
            using transform_inclusive_scan_type
              = ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation,
              initial_value, environment);
          }
        }; // struct transform_inclusive_scan< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >

        template <typename LocalState_>
        struct transform_inclusive_scan_self;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct transform_inclusive_scan_self<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <
            typename ParallelPolicy,
            typename BinaryOperation, typename UnaryOperation>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            yampi::environment const& environment)
          {
            using transform_inclusive_scan_self_type
              = ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, environment);
          }

          template <
            typename ParallelPolicy,
            typename BinaryOperation, typename UnaryOperation>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex const initial_value, yampi::environment const& environment)
          {
            using transform_inclusive_scan_self_type
              = ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, initial_value, environment);
          }
        }; // struct transform_inclusive_scan_self< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >

        template <typename LocalState_>
        struct upper_bound;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct upper_bound<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <typename Compare>
          static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            Complex const& value, Compare compare, yampi::environment const& environment)
          {
            using upper_bound_type
              = ::ket::mpi::state_detail::upper_bound<num_page_qubits>;
            return upper_bound_type::call(local_state, value, compare, environment);
          }
        }; // struct upper_bound< ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
      } // namespace dispatch
    } // namespace utility
  } // namespace mpi
} // namespace ket


# undef KET_is_nothrow_swappable

#endif // KET_MPI_STATE_HPP
