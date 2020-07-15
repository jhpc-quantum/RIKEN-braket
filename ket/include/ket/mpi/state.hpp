#ifndef KET_MPI_STATE_HPP
# define KET_MPI_STATE_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <iterator>
# include <vector>
# include <memory>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/integral_constant.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#   include <initializer_list>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/cstdint.hpp>

# ifdef BOOST_NO_CXX11_ADDRESSOF
#   include <boost/core/addressof.hpp>
# endif

# include <boost/utility.hpp> // boost::prior

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

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_true_type std::true_type
# else
#   define KET_true_type boost::true_type
# endif

# if __cplusplus >= 201703L
#   define KET_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define KET_is_nothrow_swappable boost::is_nothrow_swappable
# endif

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define KET_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T
#   define KET_FORWARD_OR_COPY(T, x) x
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define KET_addressof std::addressof
# else
#   define KET_addressof boost::addressof
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
        BOOST_CONSTEXPR state_iterator() BOOST_NOEXCEPT
          : state_ptr_(nullptr), index_()
        { }

        BOOST_CONSTEXPR state_iterator(State& state, int const index) BOOST_NOEXCEPT
          : state_ptr_(KET_addressof(state)), index_(index)
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

        void swap(state_iterator& other) BOOST_NOEXCEPT
        {
          using std::swap;
          swap(state_ptr_, other.state_ptr_);
          swap(index_, other.index_);
        }
      };

      template <typename State>
      inline void swap(
        ::ket::mpi::state_detail::state_iterator<State>& lhs,
        ::ket::mpi::state_detail::state_iterator<State>& rhs) BOOST_NOEXCEPT
      { lhs.swap(rhs); }
    } // namespace state_detail


    // NOTE: Assuming size() % (1 << num_page_qubits) == 0
    template <typename Complex, int num_page_qubits, typename Allocator>
    class state
    {
     public:
      BOOST_STATIC_CONSTEXPR std::size_t num_pages = 1u << num_page_qubits;

      typedef Complex value_type;
      typedef typename Allocator::template rebind<value_type>::other allocator_type;

     private:
      typedef std::vector<value_type, allocator_type> data_type;
      data_type data_;

     public:
      typedef
        boost::iterator_range<typename ::ket::utility::meta::iterator_of<data_type>::type>
        page_range_type;

     public:
      std::size_t num_local_qubits_;
      KET_array<page_range_type, num_pages> page_ranges_;
      page_range_type buffer_range_;

     public:
      typedef typename data_type::size_type size_type;
      typedef typename data_type::difference_type difference_type;
      typedef typename data_type::reference reference;
      typedef typename data_type::const_reference const_reference;
      typedef typename data_type::pointer pointer;
      typedef typename data_type::const_pointer const_pointer;
      typedef ::ket::mpi::state_detail::state_iterator<state> iterator;
      typedef ::ket::mpi::state_detail::state_iterator<state const> const_iterator;
      typedef std::reverse_iterator<iterator> reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

      state() BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( allocator_type() )) ))
        : data_(allocator_type()), num_local_qubits_(), page_ranges_(), buffer_range_()
      { }
      explicit state(allocator_type const& allocator) BOOST_NOEXCEPT
        : data_(allocator), num_local_qubits_(), page_ranges_(), buffer_range_()
      { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~state() BOOST_NOEXCEPT = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      state(state&&) = default;
      state& operator=(state&&) = default;
#   endif
# endif

      state(state const& other, allocator_type const& allocator)
        : data_(other.data_, allocator),
          num_local_qubits_(other.num_local_qubits_),
          page_ranges_(other.page_ranges_),
          buffer_range_(other.buffer_range_)
      { }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      state(state&& other, allocator_type const& allocator)
        : data_(std::move(other.data_), allocator),
          num_local_qubits_(std::move(other.num_local_qubits_)),
          page_ranges_(std::move(other.page_ranges_)),
          buffer_range_(std::move(other.buffer_range_))
      { }
# endif

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_(generate_initial_data(initializer_list, allocator)),
          num_local_qubits_(::ket::utility::integer_log2(initializer_list.size())),
          page_ranges_(generate_initial_page_ranges(data_)),
          buffer_range_(generate_initial_buffer_range(data_))
      { }
# endif


      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_(generate_initial_data(
            ::ket::mpi::utility::policy::make_general_mpi(),
            num_local_qubits, initial_integer, permutation, communicator, environment)),
          num_local_qubits_(num_local_qubits),
          page_ranges_(generate_initial_page_ranges(data_)),
          buffer_range_(generate_initial_buffer_range(data_))
      { }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_(generate_initial_data(
            mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment)),
          num_local_qubits_(num_local_qubits),
          page_ranges_(generate_initial_page_ranges(data_)),
          buffer_range_(generate_initial_buffer_range(data_))
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
        return static_cast<BitInteger>(permutated_qubit) >= num_local_qubits_-num_page_qubits
          and static_cast<BitInteger>(permutated_qubit) < num_local_qubits_;
      }

      std::size_t num_local_qubits() const { return num_local_qubits_; }


      bool operator==(state const& other) const
      { return data_ == other.data_; }
      bool operator<(state const& other) const
      { return data_ < other.data_; }


      // Element access
      reference at(size_type const index)
      {
        return data_.at(
          (::ket::utility::begin(page_ranges_[get_page_id(index)])-data_.begin())+get_index_in_page(index));
      }

      const_reference at(size_type const index) const
      {
        return data_.at(
          (::ket::utility::begin(page_ranges_[get_page_id(index)])-data_.begin())+get_index_in_page(index));
      }

      reference operator[](size_type const index)
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));
        return ::ket::utility::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      const_reference operator[](size_type const index) const
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));
        return ::ket::utility::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      reference front() { return *::ket::utility::begin(page_ranges_[get_page_id(0u)]); }
      const_reference front() const { return *::ket::utility::begin(page_ranges_[get_page_id(0u)]); }

      reference back() { return *--::ket::utility::end(page_ranges_[get_page_id((1u << num_local_qubits_)-1u)]); }
      const_reference back() const { return *--::ket::utility::end(page_ranges_[get_page_id((1u << num_local_qubits_)-1u)]); }


      // Iterators
      iterator begin() BOOST_NOEXCEPT { return iterator(*this, 0); }
      const_iterator begin() const BOOST_NOEXCEPT { return const_iterator(*this, 0); }
      const_iterator cbegin() const BOOST_NOEXCEPT { return const_iterator(*this, 0); }
      iterator end() BOOST_NOEXCEPT
      { return iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      const_iterator end() const BOOST_NOEXCEPT
      { return const_iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      const_iterator cend() const BOOST_NOEXCEPT
      { return const_iterator(*this, static_cast<int>(1u << num_local_qubits_)); }
      reverse_iterator rbegin() BOOST_NOEXCEPT { return reverse_iterator(end()); }
      const_reverse_iterator rbegin() const BOOST_NOEXCEPT { return const_reverse_iterator(end()); }
      const_reverse_iterator crbegin() const BOOST_NOEXCEPT { return const_reverse_iterator(cend()); }
      reverse_iterator rend() BOOST_NOEXCEPT { return reverse_iterator(begin()); }
      const_reverse_iterator rend() const BOOST_NOEXCEPT { return const_reverse_iterator(begin()); }
      const_reverse_iterator crend() const BOOST_NOEXCEPT { return const_reverse_iterator(cbegin()); }


      // Capacity
      bool empty() const BOOST_NOEXCEPT { return data_.empty(); }
      size_type size() const BOOST_NOEXCEPT { return data_.size()-boost::size(buffer_range_); }
      size_type max_size() const BOOST_NOEXCEPT { return data_.max_size()-boost::size(buffer_range_); }
      void reserve(size_type const new_capacity) { data_.reserve(new_capacity+boost::size(buffer_range_)); }
      size_type capacity() const BOOST_NOEXCEPT { return data_.capacity()-boost::size(buffer_range_); }
      void shrink_to_fit() { data_.shrink_to_fit(); }


      // Modifiers
      void swap(state& other)
        BOOST_NOEXCEPT_IF((
          KET_is_nothrow_swappable<data_type>::value
          and KET_is_nothrow_swappable<std::size_t>::value
          and KET_is_nothrow_swappable< KET_array<page_range_type, num_pages> >::value
          and KET_is_nothrow_swappable<page_range_type>::value ))
      {
        using std::swap;
        swap(data_, other.data_);

        swap(num_local_qubits_, other.num_local_qubits_);
        swap(page_ranges_, other.page_ranges_);
        swap(buffer_range_, other.buffer_range_);
      }


     private:
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
      data_type generate_initial_data(
        std::initializer_list<value_type> initializer_list,
        allocator_type const& allocator) const
      {
        data_type result(allocator);

        std::size_t const state_size = initializer_list.size();
        std::size_t const result_size = state_size + state_size / num_pages;

        assert(state_size % num_pages == 0);

        result.reserve(result_size);
        result.assign(initializer_list);
        result.resize(result_size);
        return result;
      }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST

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
        data_type result;

        std::size_t const state_size = ::ket::utility::integer_exp2<std::size_t>(num_local_qubits);
        std::size_t const result_size = state_size + state_size / num_pages;

        assert(state_size % num_pages == 0);

        result.reserve(result_size);
        result.assign(state_size, value_type(0));

        using ::ket::mpi::permutate_bits;
        std::pair<yampi::rank, StateInteger> const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, result, permutate_bits(permutation, initial_integer));

        if (communicator.rank(environment) == rank_index.first)
          result[rank_index.second] = value_type(1);

        result.resize(result_size);
        return result;
      }


      KET_array<page_range_type, num_pages>
      generate_initial_page_ranges(data_type& data) const
      {
        assert(data.size() % (num_pages+1u) == 0u);
        size_type const page_size = data.size() / (num_pages+1u);

        KET_array<page_range_type, num_pages> result;
        for (std::size_t page_id = 0u; page_id < num_pages; ++page_id)
        {
          result[page_id]
            = boost::make_iterator_range(
                ::ket::utility::begin(data)+page_id*page_size,
                ::ket::utility::begin(data)+(page_id+1u)*page_size);
        }

        return result;
      }

      page_range_type generate_initial_buffer_range(data_type& data) const
      {
        assert(data.size() % (num_pages+1u) == 0u);
        size_type const page_size = data.size() / (num_pages+1u);

        return boost::make_iterator_range(
          ::ket::utility::begin(data)+num_pages*page_size,
          ::ket::utility::begin(data)+(num_pages+1u)*page_size);
      }


     public:
      size_type get_page_id(size_type const index) const
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));

        std::size_t const num_qubits_in_page = num_local_qubits_-num_page_qubits;
        return (((num_pages-1u) << num_qubits_in_page) bitand index) >> num_qubits_in_page;
      }

      size_type get_index_in_page(size_type const index) const
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));

        std::size_t const num_qubits_in_page = num_local_qubits_-num_page_qubits;
        return (compl ((num_pages-1u) << num_qubits_in_page)) bitand index;
      }
    };


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

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        BitInteger const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits);
        boost::tuple<qubit_type, qubit_type> const minmax_qubits
          = boost::minmax(permutated_qubit1, permutated_qubit2);
        using boost::get;
        StateInteger const lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(get<0u>(minmax_qubits)-static_cast<qubit_type>(num_nonpage_qubits))
            - static_cast<StateInteger>(1u);
        StateInteger const middle_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(
               get<1u>(minmax_qubits)-static_cast<qubit_type>(num_nonpage_qubits+1u))
             - static_cast<StateInteger>(1u))
            xor lower_bits_mask;
        StateInteger const upper_bits_mask
          = compl (lower_bits_mask bitor middle_bits_mask);


        for (StateInteger value_wo_qubits = 0u;
             value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_page_qubits-2));
             ++value_wo_qubits)
        {
          StateInteger const base_page_index
            = ((value_wo_qubits bitand upper_bits_mask) << 2u)
              bitor ((value_wo_qubits bitand middle_bits_mask) << 1u)
              bitor (value_wo_qubits bitand lower_bits_mask);
          StateInteger const page_index1
            = base_page_index
              bitor (static_cast<StateInteger>(1u)
                     << (permutated_qubit1-static_cast<qubit_type>(num_nonpage_qubits)));
          StateInteger const page_index2
            = base_page_index
              bitor (static_cast<StateInteger>(1u)
                     << (permutated_qubit2-static_cast<qubit_type>(num_nonpage_qubits)));

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
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        BitInteger const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits);
        boost::tuple<qubit_type, qubit_type> const minmax_qubits
          = boost::minmax(permutated_qubit1, permutated_qubit2);
        using boost::get;
        StateInteger const nonpage_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(get<0u>(minmax_qubits))
            - static_cast<StateInteger>(1u);
        StateInteger const nonpage_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(num_nonpage_qubits-1u)-static_cast<StateInteger>(1u))
            xor nonpage_lower_bits_mask;
        StateInteger const page_lower_bits_mask
          = ::ket::utility::integer_exp2<StateInteger>(get<1u>(minmax_qubits)-static_cast<qubit_type>(num_nonpage_qubits))
            - static_cast<StateInteger>(1u);
        StateInteger const page_upper_bits_mask
          = (::ket::utility::integer_exp2<StateInteger>(num_page_qubits-1u)-static_cast<StateInteger>(1u))
            xor page_lower_bits_mask;


        for (StateInteger page_value_wo_qubits = 0u;
             page_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_page_qubits-1u));
             ++page_value_wo_qubits)
        {
          StateInteger const page_index0
            = ((page_value_wo_qubits bitand page_upper_bits_mask) << 1u)
              bitor (page_value_wo_qubits bitand page_lower_bits_mask);
          StateInteger const page_index1
            = (static_cast<StateInteger>(1u) << (get<1u>(minmax_qubits)-static_cast<qubit_type>(num_nonpage_qubits)))
              bitor page_index0;

          for (StateInteger nonpage_value_wo_qubits = 0u;
               nonpage_value_wo_qubits < ::ket::utility::integer_exp2<StateInteger>(static_cast<StateInteger>(num_nonpage_qubits-1u));
               ++nonpage_value_wo_qubits)
          {
            StateInteger const nonpage_index0
              = ((nonpage_value_wo_qubits bitand nonpage_upper_bits_mask) << 1u)
                bitor (nonpage_value_wo_qubits bitand nonpage_lower_bits_mask);
            StateInteger const nonpage_index1
              = nonpage_index0 bitor (static_cast<StateInteger>(1u) << get<0u>(minmax_qubits));

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
      };

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
      };


      template <int num_page_qubits>
      struct interchange_qubits
      {
        static_assert(
          num_page_qubits >= 1,
          "num_page_qubits should be at least 1 if using this function");

# ifdef BOOST_NO_CXX11_LAMBDAS
        class yampi_swap
        {
          yampi::rank target_rank_;
          yampi::communicator const& communicator_;
          yampi::environment const& environment_;

         public:
          yampi_swap(
            yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
            : target_rank_(target_rank), communicator_(communicator), environment_(environment)
          { }

          template <typename PageIterator>
          void operator()(
            PageIterator const first, PageIterator const last,
            PageIterator const buffer_first, PageIterator const buffer_last) const
          {
            yampi::algorithm::swap(
              yampi::ignore_status(),
              yampi::make_buffer(first, last),
              yampi::make_buffer(buffer_first, buffer_last),
              target_rank_, communicator_, environment_);
          }
        }; // class yampi_swap

        template <typename DerivedDatatype>
        class yampi_swap_with_datatype
        {
          yampi::datatype_base<DerivedDatatype> const& datatype_;
          yampi::rank target_rank_;
          yampi::communicator const& communicator_;
          yampi::environment const& environment_;

         public:
          yampi_swap_with_datatype(
            yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
            : datatype_(datatype), target_rank_(target_rank), communicator_(communicator), environment_(environment)
          { }

          template <typename PageIterator>
          void operator()(
            PageIterator const first, PageIterator const last,
            PageIterator const buffer_first, PageIterator const buffer_last) const
          {
            yampi::algorithm::swap(
              yampi::ignore_status(),
              yampi::make_buffer(first, last, datatype_),
              yampi::make_buffer(buffer_first, buffer_last, datatype_),
              target_rank_, communicator_, environment_);
          }
        }; // class yampi_swap_with_datatype<DerivedDatatype>
# endif // BOOST_NO_CXX11_LAMBDAS

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
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::page_range_type
            page_range_type;
          typedef
            typename ::ket::utility::meta::iterator_of<page_range_type>::type
            page_iterator;
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
# else // BOOST_NO_CXX11_LAMBDAS
          do_call(
            local_state, source_local_first_index, source_local_last_index,
            yampi_swap(target_rank, communicator, environment));
# endif // BOOST_NO_CXX11_LAMBDAS
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
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::page_range_type
            page_range_type;
          typedef
            typename ::ket::utility::meta::iterator_of<page_range_type>::type
            page_iterator;
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
# else // BOOST_NO_CXX11_LAMBDAS
          do_call(
            local_state, source_local_first_index, source_local_last_index,
            yampi_swap_with_datatype<DerivedDatatype>(datatype, target_rank, communicator, environment));
# endif // BOOST_NO_CXX11_LAMBDAS
        }

       private:
        template <
          typename Allocator, typename Complex, typename StateInteger, typename Function>
        static void do_call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          StateInteger const source_local_first_index, StateInteger const source_local_last_index,
          KET_RVALUE_REFERENCE_OR_COPY(Function) yampi_swap)
        {
          assert(source_local_last_index >= source_local_first_index);

          typedef
            typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::size_type
            size_type;
          size_type const page_front_id = local_state.get_page_id(source_local_first_index);
          size_type const page_back_id = local_state.get_page_id(source_local_last_index-1u);

          for (std::size_t page_id = page_front_id; page_id <= page_back_id; ++page_id)
          {
            typedef
              typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::page_range_type
              page_range_type;
            page_range_type page_range = local_state.page_range(page_id);
            size_type const page_size = boost::size(page_range);

            typedef
              typename ::ket::utility::meta::iterator_of<page_range_type>::type
              page_iterator;
            page_iterator const page_first = ::ket::utility::begin(page_range);
            page_iterator const page_last = ::ket::utility::end(page_range);
            page_iterator const buffer_first = ::ket::utility::begin(local_state.buffer_range());

            StateInteger const first_index
              = page_id == page_front_id
                ? static_cast<StateInteger>(
                    local_state.get_index_in_page(source_local_first_index))
                : static_cast<StateInteger>(0u);
            StateInteger const last_index
              = page_id == page_back_id
                ? static_cast<StateInteger>(
                    local_state.get_index_in_page(source_local_last_index-1u)+1u)
                : static_cast<StateInteger>(page_size);

            page_iterator const the_first = page_first + first_index;
            page_iterator const the_last = page_first + last_index;
            page_iterator const the_buffer_first = buffer_first + first_index;
            page_iterator const the_buffer_last = buffer_first + last_index;

            std::copy(page_first, the_first, buffer_first);
            std::copy(the_last, page_last, the_buffer_last);

            yampi_swap(the_first, the_last, the_buffer_first, the_buffer_last);

            local_state.swap_buffer_and_page(page_id);
          }
        }
      };


# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename Qubit>
      struct is_page_qubit
      {
       private:
        Qubit least_significant_page_qubit_;

       public:
        typedef bool result_type;

        is_page_qubit(Qubit const least_significant_page_qubit)
          : least_significant_page_qubit_(least_significant_page_qubit)
        { }

        bool operator()(Qubit const qubit) const { return qubit >= least_significant_page_qubit_; }
      };

      template <typename Qubit>
      ::ket::mpi::state_detail::is_page_qubit<Qubit> make_is_page_qubit(
        Qubit const least_significant_page_qubit)
      { return ::ket::mpi::state_detail::is_page_qubit<Qubit>(least_significant_page_qubit); }
# endif // BOOST_NO_CXX11_LAMBDAS

      template <int num_page_qubits>
      struct for_each_local_range
      {
        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, num_page_qubits, Allocator>& call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          // Gates should not be on page qubits
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
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
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          // Gates should not be on page qubits
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
            function(
              ::ket::utility::begin(local_state.page_range(page_id)),
              ::ket::utility::end(local_state.page_range(page_id)));
          return local_state;
        }
      };


      template <int num_page_qubits>
      struct transform_inclusive_scan
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
#   ifdef BOOST_NO_CXX11_LAMBDAS
        template <
          typename PageIterator, typename ForwardIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation>
        class loop_body_in_process_in_execute
        {
          std::size_t page_id_;
          PageIterator first_;
          ForwardIterator& d_iter_;
          bool& is_called_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;

         public:
          loop_body_in_process_in_execute(
            std::size_t page_id, PageIterator first, ForwardIterator& d_iter,
            bool& is_called, unsigned int num_threads, PartialSums& partial_sums,
            BinaryOperation binary_operation, UnaryOperation unary_operation)
            : page_id_(page_id),
              first_(first),
              d_iter_(d_iter),
              is_called_(is_called),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation)
          { }

          typedef
            typename std::iterator_traits<PageIterator>::difference_type
            difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (not is_called_)
            {
              std::advance(d_iter_, n);
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = unary_operation_(first_[n]);
              is_called_ = true;
            }
            else
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = binary_operation_(
                    partial_sums_[num_threads_ * page_id_ + thread_index],
                    unary_operation_(first_[n]));

            *d_iter_++ = partial_sums_[num_threads_ * page_id_ + thread_index];
          }
        };

        template <
          typename PageIterator, typename ForwardIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation>
        static loop_body_in_process_in_execute<PageIterator, ForwardIterator, PartialSums, BinaryOperation, UnaryOperation>
        make_loop_body_in_process_in_execute(
          std::size_t page_id, PageIterator first, ForwardIterator& d_iter,
          bool& is_called, unsigned int num_threads, PartialSums& partial_sums,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          typedef
            loop_body_in_process_in_execute<PageIterator, ForwardIterator, PartialSums, BinaryOperation, UnaryOperation>
            result_type;
          return result_type(
            page_id, first, d_iter, is_called, num_threads, partial_sums, binary_operation, unary_operation);
        }


        template <
          typename PageIterator, typename ForwardIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        class loop_body_in_process_in_execute_with_initial_value
        {
          std::size_t page_id_;
          PageIterator first_;
          ForwardIterator& d_iter_;
          bool& is_called_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;
          Complex initial_value_;

         public:
          loop_body_in_process_in_execute_with_initial_value(
            std::size_t page_id, PageIterator first, ForwardIterator& d_iter,
            bool& is_called, unsigned int num_threads, PartialSums& partial_sums,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex initial_value)
            : page_id_(page_id),
              first_(first),
              d_iter_(d_iter),
              is_called_(is_called),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation),
              initial_value_(initial_value)
          { }

          typedef
            typename std::iterator_traits<PageIterator>::difference_type
            difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (not is_called_)
            {
              std::advance(d_iter_, n);
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = thread_index == 0
                  ? binary_operation_(initial_value_, unary_operation_(first_[n]))
                  : unary_operation_(first_[n]);
              is_called_ = true;
            }
            else
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = binary_operation_(
                    partial_sums_[num_threads_ * page_id_ + thread_index],
                    unary_operation_(first_[n]));

            *d_iter_++ = partial_sums_[num_threads_ * page_id_ + thread_index];
          }
        };

        template <
          typename PageIterator, typename ForwardIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static loop_body_in_process_in_execute_with_initial_value<PageIterator, ForwardIterator, PartialSums, BinaryOperation, UnaryOperation, Complex>
        make_loop_body_in_process_in_execute_with_initial_value(
          std::size_t page_id, PageIterator first, ForwardIterator& d_iter,
          bool& is_called, unsigned int num_threads, PartialSums& partial_sums,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Complex initial_value)
        {
          typedef
            loop_body_in_process_in_execute_with_initial_value<PageIterator, ForwardIterator, PartialSums, BinaryOperation, UnaryOperation, Complex>
            result_type;
          return result_type(
            page_id, first, d_iter, is_called, num_threads, partial_sums, binary_operation, unary_operation, initial_value);
        }


        template <typename PartialSums, typename BinaryOperation>
        class process_in_single_execute
        {
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;

         public:
          process_in_single_execute(PartialSums& partial_sums, BinaryOperation binary_operation)
            : partial_sums_(partial_sums), binary_operation_(binary_operation)
          { }

          void operator()()
          {
            std::partial_sum(
              ::ket::utility::begin(partial_sums_), ::ket::utility::end(partial_sums_),
              ::ket::utility::begin(partial_sums_), binary_operation_);
          }
        };

        template <typename PartialSums, typename BinaryOperation>
        static process_in_single_execute<PartialSums, BinaryOperation>
        make_process_in_single_execute(PartialSums& partial_sums, BinaryOperation binary_operation)
        { return process_in_single_execute<PartialSums, BinaryOperation>(partial_sums, binary_operation); }


        template <typename ForwardIterator, typename PartialSums, typename BinaryOperation>
        class loop_body_in_post_process
        {
          std::size_t page_id_;
          ForwardIterator& d_iter_;
          bool& is_called_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;

         public:
          loop_body_in_post_process(
            std::size_t page_id, ForwardIterator& d_iter, bool& is_called,
            unsigned int num_threads, PartialSums& partial_sums, BinaryOperation binary_operation)
            : page_id_(page_id),
              d_iter_(d_iter),
              is_called_(is_called),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation)
          { }

          typedef typename PartialSums::difference_type difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (thread_index == 0u and page_id_ == 0u)
              return;

            if (not is_called_)
            {
              std::advance(d_iter_, n);
              is_called_ = true;
            }

            *d_iter_
              = binary_operation_(
                  partial_sums_[num_threads_ * page_id_ + thread_index - 1u],
                  *d_iter_);
            ++d_iter_;
          }
        };

        template <typename ForwardIterator, typename PartialSums, typename BinaryOperation>
        static loop_body_in_post_process<ForwardIterator, PartialSums, BinaryOperation>
        make_loop_body_in_post_process(
          std::size_t page_id, ForwardIterator& d_iter, bool& is_called,
          unsigned int num_threads, PartialSums& partial_sums, BinaryOperation binary_operation)
        {
          typedef
            loop_body_in_post_process<ForwardIterator, PartialSums, BinaryOperation>
            result_type;
          return result_type(page_id, d_iter, is_called, num_threads, partial_sums, binary_operation);
        }
#   endif // BOOST_NO_CXX11_LAMBDAS


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
            : num_threads_(num_threads),
              partial_sums_(partial_sums),
              parallel_policy_(parallel_policy),
              local_state_(local_state),
              d_first_(d_first),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation)
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            ForwardIterator d_page_first = d_first_;

            for (std::size_t page_id = 0u; page_id < LocalState::num_pages; ++page_id)
            {
              typedef typename LocalState::page_range_type page_range_type;
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const first
                = ::ket::utility::begin(local_state_.page_range(page_id));
              ForwardIterator d_iter = d_page_first;
              bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
              typedef
                typename std::iterator_traits<page_iterator>::difference_type
                difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                make_loop_body_in_process_in_execute(
                  page_id, first, d_iter, is_called, num_threads_, partial_sums_,
                  binary_operation_, unary_operation_));
#     endif // BOOST_NO_CXX11_LAMBDAS

              std::advance(d_page_first, boost::size(local_state_.page_range(page_id)));
            }

            post_process(
              parallel_policy_, local_state_, d_first_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        };

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
          typedef
            process_in_execute<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation>
            result_type;
          return result_type(
            num_threads, partial_sums, parallel_policy, local_state, d_first, binary_operation, unary_operation);
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
            : num_threads_(num_threads),
              partial_sums_(partial_sums),
              parallel_policy_(parallel_policy),
              local_state_(local_state),
              d_first_(d_first),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation),
              initial_value_(initial_value)
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            ForwardIterator d_page_first = d_first_;

            for (std::size_t page_id = 0u; page_id < LocalState::num_pages; ++page_id)
            {
              typedef typename LocalState::page_range_type page_range_type;
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const first
                = ::ket::utility::begin(local_state_.page_range(page_id));
              ForwardIterator d_iter = d_page_first;
              bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
              typedef
                typename std::iterator_traits<page_iterator>::difference_type
                difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                make_loop_body_in_process_in_execute_with_initial_value(
                  page_id, first, d_iter, is_called, num_threads_, partial_sums_,
                  binary_operation_, unary_operation_, initial_value_));
#     endif // BOOST_NO_CXX11_LAMBDAS

              std::advance(d_page_first, boost::size(local_state_.page_range(page_id)));
            }

            post_process(
              parallel_policy_, local_state_, d_first_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        };

        template <
          typename PartialSums, typename ParallelPolicy,
          typename LocalState, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation, Complex>
        make_process_in_execute_with_initial_value(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy,
          LocalState& local_state, ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Complex initial_value)
        {
          typedef
            process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, ForwardIterator, BinaryOperation, UnaryOperation, Complex>
            result_type;
          return result_type(
            num_threads, partial_sums, parallel_policy, local_state, d_first, binary_operation, unary_operation, initial_value);
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
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          std::vector<Complex> partial_sums(num_threads * local_state_type::num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              ForwardIterator d_page_first = d_first;

              for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
              {
                typedef typename local_state_type::page_range_type page_range_type;
                typedef typename boost::range_iterator<page_range_type>::type page_iterator;
                page_iterator const first
                  = ::ket::utility::begin(local_state.page_range(page_id));
                ForwardIterator d_iter = d_page_first;
                bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
                typedef
                  typename std::iterator_traits<page_iterator>::difference_type
                  difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
                ::ket::utility::loop_n_in_execute(
                  parallel_policy_,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  make_loop_body_in_process_in_execute(
                    page_id, first, d_iter, is_called, num_threads, partial_sums,
                    binary_operation, unary_operation));
#     endif // BOOST_NO_CXX11_LAMBDAS

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
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          std::vector<Complex> partial_sums(num_threads * local_state_type::num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, d_first, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              ForwardIterator d_page_first = d_first;

              for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
              {
                typedef typename local_state_type::page_range_type page_range_type;
                typedef typename boost::range_iterator<page_range_type>::type page_iterator;
                page_iterator const first
                  = ::ket::utility::begin(local_state.page_range(page_id));
                ForwardIterator d_iter = d_page_first;
                bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
                typedef
                  typename std::iterator_traits<page_iterator>::difference_type
                  difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  make_loop_body_in_process_in_execute_with_initial_value(
                    page_id, first, d_iter, is_called, num_threads, partial_sums,
                    binary_operation, unary_operation, initial_value));
#     endif // BOOST_NO_CXX11_LAMBDAS

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

#   ifndef BOOST_NO_CXX11_LAMBDAS
          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              std::partial_sum(
                ::ket::utility::begin(partial_sums), ::ket::utility::end(partial_sums),
                ::ket::utility::begin(partial_sums), binary_operation);
            });
#   else // BOOST_NO_CXX11_LAMBDAS
          ::ket::utility::single_execute(
            parallel_policy, executor,
            make_process_in_single_execute(partial_sums, binary_operation));
#   endif // BOOST_NO_CXX11_LAMBDAS

          ForwardIterator d_page_first = d_first;

          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ForwardIterator d_iter = d_page_first;
            bool is_called = false;

#   ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename local_state_type::page_range_type page_range_type;
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            typedef
              typename std::iterator_traits<page_iterator>::difference_type
              difference_type;
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
#   else // BOOST_NO_CXX11_LAMBDAS
            ::ket::utility::loop_n_in_execute(
              parallel_policy,
              boost::size(local_state.page_range(page_id)), thread_index,
              make_loop_body_in_post_process(
                page_id, d_iter, is_called, num_threads, partial_sums, binary_operation));
#   endif // BOOST_NO_CXX11_LAMBDAS

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
          ForwardIterator prev_d_first = d_first;
          d_first
            = ::ket::utility::ranges::transform_inclusive_scan(
                parallel_policy,
                local_state.page_range(0u), d_first, binary_operation, unary_operation);
          std::advance(prev_d_first, boost::size(local_state.page_range(0u))-1);
          Complex partial_sum = *prev_d_first;

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 1u; page_id < local_state_type::num_pages; ++page_id)
          {
            prev_d_first = d_first;
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, boost::size(local_state.page_range(page_id))-1);
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
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ForwardIterator prev_d_first = d_first;
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, boost::size(local_state.page_range(page_id))-1);
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
          Complex partial_sum = *boost::prior(d_first);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 1u; page_id < local_state_type::num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(d_first);
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
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::ranges::transform_inclusive_scan(
                  parallel_policy,
                  local_state.page_range(page_id), d_first,
                  binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(d_first);
          }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      };


      template <int num_page_qubits>
      struct transform_inclusive_scan_self
      {
# ifdef KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
#   ifdef BOOST_NO_CXX11_LAMBDAS
        template <
          typename PageIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation>
        class loop_body_in_process_in_execute
        {
          std::size_t page_id_;
          PageIterator first_;
          bool& is_called_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;

         public:
          loop_body_in_process_in_execute(
            std::size_t page_id, PageIterator first, bool& is_called,
            unsigned int num_threads, PartialSums& partial_sums,
            BinaryOperation binary_operation, UnaryOperation unary_operation)
            : page_id_(page_id),
              first_(first),
              is_called_(is_called),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation)
          { }

          typedef
            typename std::iterator_traits<PageIterator>::difference_type
            difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (not is_called_)
            {
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = unary_operation_(first_[n]);
              is_called_ = true;
            }
            else
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = binary_operation_(
                    partial_sums_[num_threads_ * page_id_ + thread_index],
                    unary_operation_(first_[n]));

            first_[n] = partial_sums_[num_threads_ * page_id_ + thread_index];
          }
        };

        template <
          typename PageIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation>
        static loop_body_in_process_in_execute<PageIterator, PartialSums, BinaryOperation, UnaryOperation>
        make_loop_body_in_process_in_execute(
          std::size_t page_id, PageIterator first, bool& is_called,
          unsigned int num_threads, PartialSums& partial_sums,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          typedef
            loop_body_in_process_in_execute<PageIterator, PartialSums, BinaryOperation, UnaryOperation>
            result_type;
          return result_type(
            page_id, first, is_called, num_threads, partial_sums, binary_operation, unary_operation);
        }


        template <
          typename PageIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        class loop_body_in_process_in_execute_with_initial_value
        {
          std::size_t page_id_;
          PageIterator first_;
          bool& is_called_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;
          UnaryOperation unary_operation_;
          Complex initial_value_;

         public:
          loop_body_in_process_in_execute_with_initial_value(
            std::size_t page_id, PageIterator first, bool& is_called,
            unsigned int num_threads, PartialSums& partial_sums,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Complex initial_value)
            : page_id_(page_id),
              first_(first),
              is_called_(is_called),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation),
              initial_value_(initial_value)
          { }

          typedef
            typename std::iterator_traits<PageIterator>::difference_type
            difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (not is_called_)
            {
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = page_id_ == 0 && thread_index == 0
                  ? binary_operation_(initial_value_, unary_operation_(first_[n]))
                  : unary_operation_(first_[n]);
              is_called_ = true;
            }
            else
              partial_sums_[num_threads_ * page_id_ + thread_index]
                = binary_operation_(
                    partial_sums_[num_threads_ * page_id_ + thread_index],
                    unary_operation_(first_[n]));

            first_[n] = partial_sums_[num_threads_ * page_id_ + thread_index];
          }
        };

        template <
          typename PageIterator, typename PartialSums,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static loop_body_in_process_in_execute_with_initial_value<PageIterator, PartialSums, BinaryOperation, UnaryOperation, Complex>
        make_loop_body_in_process_in_execute_with_initial_value(
          std::size_t page_id, PageIterator first, bool& is_called,
          unsigned int num_threads, PartialSums& partial_sums,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Complex initial_value)
        {
          typedef
            loop_body_in_process_in_execute_with_initial_value<PageIterator, PartialSums, BinaryOperation, UnaryOperation, Complex>
            result_type;
          return result_type(
            page_id, first, is_called, num_threads, partial_sums, binary_operation, unary_operation, initial_value);
        }


        template <typename PartialSums, typename BinaryOperation>
        class process_in_single_execute
        {
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;

         public:
          process_in_single_execute(PartialSums& partial_sums, BinaryOperation binary_operation)
            : partial_sums_(partial_sums), binary_operation_(binary_operation)
          { }

          void operator()()
          {
            std::partial_sum(
              ::ket::utility::begin(partial_sums_), ::ket::utility::end(partial_sums_),
              ::ket::utility::begin(partial_sums_), binary_operation_);
          }
        };

        template <typename PartialSums, typename BinaryOperation>
        static process_in_single_execute<PartialSums, BinaryOperation>
        make_process_in_single_execute(PartialSums& partial_sums, BinaryOperation binary_operation)
        { return process_in_single_execute<PartialSums, BinaryOperation>(partial_sums, binary_operation); }


        template <typename PageIterator, typename PartialSums, typename BinaryOperation>
        class loop_body_in_post_process
        {
          std::size_t page_id_;
          PageIterator first_;
          unsigned int num_threads_;
          PartialSums& partial_sums_;
          BinaryOperation binary_operation_;

         public:
          loop_body_in_post_process(
            std::size_t page_id, PageIterator first,
            unsigned int num_threads, PartialSums& partial_sums, BinaryOperation binary_operation)
            : page_id_(page_id),
              first_(first),
              num_threads_(num_threads),
              partial_sums_(partial_sums),
              binary_operation_(binary_operation)
          { }

          typedef
            typename std::iterator_traits<PageIterator>::difference_type
            difference_type;
          void operator()(difference_type const n, int const thread_index)
          {
            if (thread_index == 0u and page_id_ == 0u)
              return;

            first_[n]
              = binary_operation_(
                  partial_sums_[num_threads_ * page_id_ + thread_index - 1u],
                  first_[n]);
          }
        };

        template <typename PageIterator, typename PartialSums, typename BinaryOperation>
        static loop_body_in_post_process<PageIterator, PartialSums, BinaryOperation>
        make_loop_body_in_post_process(
          std::size_t page_id, PageIterator first,
          unsigned int num_threads, PartialSums& partial_sums, BinaryOperation binary_operation)
        {
          typedef
            loop_body_in_post_process<PageIterator, PartialSums, BinaryOperation>
            result_type;
          return result_type(page_id, first, num_threads, partial_sums, binary_operation);
        }
#   endif // BOOST_NO_CXX11_LAMBDAS


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
            : num_threads_(num_threads),
              partial_sums_(partial_sums),
              parallel_policy_(parallel_policy),
              local_state_(local_state),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation)
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            for (std::size_t page_id = 0u; page_id < LocalState::num_pages; ++page_id)
            {
              typedef typename LocalState::page_range_type page_range_type;
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const first
                = ::ket::utility::begin(local_state_.page_range(page_id));
              bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
              typedef
                typename std::iterator_traits<page_iterator>::difference_type
                difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                make_loop_body_in_process_in_execute(
                  page_id, first, is_called, num_threads_, partial_sums_,
                  binary_operation_, unary_operation_));
#     endif // BOOST_NO_CXX11_LAMBDAS
            }

            post_process(
              parallel_policy_, local_state_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        };

        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation>
        static process_in_execute<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation>
        make_process_in_execute(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy, LocalState& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          typedef
            process_in_execute<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation>
            result_type;
          return result_type(
            num_threads, partial_sums, parallel_policy, local_state, binary_operation, unary_operation);
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
            : num_threads_(num_threads),
              partial_sums_(partial_sums),
              parallel_policy_(parallel_policy),
              local_state_(local_state),
              binary_operation_(binary_operation),
              unary_operation_(unary_operation),
              initial_value_(initial_value)
          { }

          template <typename Executor>
          void operator()(int const thread_index, Executor& executor)
          {
            for (std::size_t page_id = 0u; page_id < LocalState::num_pages; ++page_id)
            {
              typedef typename LocalState::page_range_type page_range_type;
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const first
                = ::ket::utility::begin(local_state_.page_range(page_id));
              bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
              typedef
                typename std::iterator_traits<page_iterator>::difference_type
                difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
              ::ket::utility::loop_n_in_execute(
                parallel_policy_,
                boost::size(local_state_.page_range(page_id)), thread_index,
                make_loop_body_in_process_in_execute_with_initial_value(
                  page_id, first, is_called, num_threads_, partial_sums_,
                  binary_operation_, unary_operation_, initial_value_));
#     endif // BOOST_NO_CXX11_LAMBDAS
            }

            post_process(
              parallel_policy_, local_state_, binary_operation_,
              partial_sums_, thread_index, executor);
          }
        };

        template <
          typename PartialSums, typename ParallelPolicy, typename LocalState,
          typename BinaryOperation, typename UnaryOperation, typename Complex>
        static process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation, Complex>
        make_process_in_execute_with_initial_value(
          unsigned int num_threads, PartialSums& partial_sums,
          ParallelPolicy parallel_policy, LocalState& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Complex initial_value)
        {
          typedef
            process_in_execute_with_initial_value<PartialSums, ParallelPolicy, LocalState, BinaryOperation, UnaryOperation, Complex>
            result_type;
          return result_type(
            num_threads, partial_sums, parallel_policy, local_state, binary_operation, unary_operation, initial_value);
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
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          std::vector<Complex> partial_sums(num_threads * local_state_type::num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation](
              int const thread_index, auto& executor)
            {
              for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
              {
                typedef typename local_state_type::page_range_type page_range_type;
                typedef typename boost::range_iterator<page_range_type>::type page_iterator;
                page_iterator const first
                  = ::ket::utility::begin(local_state.page_range(page_id));
                bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
                typedef
                  typename std::iterator_traits<page_iterator>::difference_type
                  difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  make_loop_body_in_process_in_execute(
                    page_id, first, is_called, num_threads, partial_sums,
                    binary_operation, unary_operation));
#     endif // BOOST_NO_CXX11_LAMBDAS
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
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          std::vector<Complex> partial_sums(num_threads * local_state_type::num_pages);

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::utility::execute(
            parallel_policy,
            [num_threads, &partial_sums,
             parallel_policy, &local_state, binary_operation, unary_operation,
             initial_value](
              int const thread_index, auto& executor)
            {
              for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
              {
                typedef typename local_state_type::page_range_type page_range_type;
                typedef typename boost::range_iterator<page_range_type>::type page_iterator;
                page_iterator const first
                  = ::ket::utility::begin(local_state.page_range(page_id));
                bool is_called = false;

#     ifndef BOOST_NO_CXX11_LAMBDAS
                typedef
                  typename std::iterator_traits<page_iterator>::difference_type
                  difference_type;
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
#     else // BOOST_NO_CXX11_LAMBDAS
                ::ket::utility::loop_n_in_execute(
                  parallel_policy,
                  boost::size(local_state.page_range(page_id)), thread_index,
                  make_loop_body_in_process_in_execute_with_initial_value(
                    page_id, first, is_called, num_threads, partial_sums,
                    binary_operation, unary_operation, initial_value));
#     endif // BOOST_NO_CXX11_LAMBDAS
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

#   ifndef BOOST_NO_CXX11_LAMBDAS
          ::ket::utility::single_execute(
            parallel_policy, executor,
            [&partial_sums, binary_operation]
            {
              std::partial_sum(
                ::ket::utility::begin(partial_sums), ::ket::utility::end(partial_sums),
                ::ket::utility::begin(partial_sums), binary_operation);
            });
#   else // BOOST_NO_CXX11_LAMBDAS
          ::ket::utility::single_execute(
            parallel_policy, executor,
            make_process_in_single_execute(partial_sums, binary_operation));
#   endif // BOOST_NO_CXX11_LAMBDAS

          unsigned int num_threads = ::ket::utility::num_threads(parallel_policy);
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            typedef typename local_state_type::page_range_type page_range_type;
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const first
              = ::ket::utility::begin(local_state.page_range(page_id));

#   ifndef BOOST_NO_CXX11_LAMBDAS
            typedef
              typename std::iterator_traits<page_iterator>::difference_type
              difference_type;
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
#   else // BOOST_NO_CXX11_LAMBDAS
            ::ket::utility::loop_n_in_execute(
              parallel_policy,
              boost::size(local_state.page_range(page_id)), thread_index,
              make_loop_body_in_post_process(
                page_id, first, num_threads, partial_sums, binary_operation));
#   endif // BOOST_NO_CXX11_LAMBDAS
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
          Complex partial_sum = *boost::prior(::ket::utility::end(local_state.page_range(0u)));

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 1u; page_id < local_state_type::num_pages; ++page_id)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state.page_range(page_id),
              ::ket::utility::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(::ket::utility::end(local_state.page_range(page_id)));
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
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ::ket::utility::ranges::transform_inclusive_scan(
              parallel_policy,
              local_state.page_range(page_id),
              ::ket::utility::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(::ket::utility::end(local_state.page_range(page_id)));
          }

          return partial_sum;
        }
# endif // KET_USE_PARALLEL_EXECUTE_FOR_TRANSFORM_INCLUSIVE_SCAN
      };


      // Usually num_page_qubits is small, so linear search for page is probably not bad.
      template <int num_page_qubits>
      struct upper_bound
      {
        template <typename Complex, typename Allocator, typename Compare>
        static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          Complex const& value, Compare compare, yampi::environment const&)
        {
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          typedef typename local_state_type::difference_type difference_type;

          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            if (not compare(value, *boost::prior(::ket::utility::end(local_state.page_range(page_id)))))
              continue;

            std::size_t index_in_page
              = std::upper_bound(
                  ::ket::utility::begin(local_state.page_range(page_id)),
                  ::ket::utility::end(local_state.page_range(page_id)),
                  value, compare)
                - ::ket::utility::begin(local_state.page_range(page_id));

            std::size_t const num_qubits_in_page
              = local_state.num_local_qubits() - num_page_qubits;
            return static_cast<difference_type>((page_id << num_qubits_in_page) bitor index_in_page);
          }

          return static_cast<difference_type>(local_state.size());
        }
      };
    } // namespace state_detail


    template <typename Complex, typename Allocator>
    class state<Complex, 0, Allocator>
    {
     public:
      typedef Complex value_type;
      typedef typename Allocator::template rebind<value_type>::other allocator_type;

      BOOST_STATIC_CONSTEXPR std::size_t num_pages = 1u;

     private:
      typedef std::vector<value_type, allocator_type> data_type;
      data_type data_;

     public:
      typedef typename data_type::size_type size_type;
      typedef typename data_type::difference_type difference_type;
      typedef typename data_type::reference reference;
      typedef typename data_type::const_reference const_reference;
      typedef typename data_type::pointer pointer;
      typedef typename data_type::const_pointer const_pointer;
      typedef typename data_type::iterator iterator;
      typedef typename data_type::const_iterator const_iterator;
      typedef typename data_type::reverse_iterator reverse_iterator;
      typedef typename data_type::const_reverse_iterator const_reverse_iterator;

      state() BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( allocator_type() )) )) : data_(allocator_type()) { }
      explicit state(allocator_type const& allocator) BOOST_NOEXCEPT : data_(allocator) { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~state() BOOST_NOEXCEPT = default;
      state(state const&) = default;
      state& operator=(state const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      state(state&&) = default;
      state& operator=(state&&) = default;
#   endif
# endif

      state(state const& other, allocator_type const& allocator)
        : data_(other.data_, allocator)
      { }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      state(state&& other, allocator_type const& allocator)
        : data_(std::move(other.data_), allocator)
      { }
# endif

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
      state(std::initializer_list<value_type> initializer_list, allocator_type const& allocator = allocator_type())
        : data_(initializer_list, allocator)
      { }
# endif


      template <typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_(generate_initial_data(
            ::ket::mpi::utility::policy::make_general_mpi(),
            num_local_qubits, initial_integer, permutation, communicator, environment))
      { }

      template <typename MpiPolicy, typename BitInteger, typename StateInteger, typename PermutationAllocator>
      state(
        MpiPolicy const mpi_policy, BitInteger const num_local_qubits, StateInteger const initial_integer,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator> const&
          permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
        : data_(generate_initial_data(
            mpi_policy, num_local_qubits, initial_integer, permutation, communicator, environment))
      { }


      bool operator==(state const& other) const
      { return data_ == other.data_; }
      bool operator<(state const& other) const
      { return data_ < other.data_; }


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
      iterator begin() BOOST_NOEXCEPT { return data_.begin(); }
      const_iterator begin() const BOOST_NOEXCEPT { return data_.begin(); }
      const_iterator cbegin() const BOOST_NOEXCEPT { return data_.cbegin(); }
      iterator end() BOOST_NOEXCEPT { return data_.end(); }
      const_iterator end() const BOOST_NOEXCEPT { return data_.end(); }
      const_iterator cend() const BOOST_NOEXCEPT { return data_.cend(); }
      reverse_iterator rbegin() BOOST_NOEXCEPT { return data_.rbegin(); }
      const_reverse_iterator rbegin() const BOOST_NOEXCEPT { return data_.rbegin(); }
      const_reverse_iterator crbegin() const BOOST_NOEXCEPT { return data_.crbegin(); }
      reverse_iterator rend() BOOST_NOEXCEPT { return data_.rend(); }
      const_reverse_iterator rend() const BOOST_NOEXCEPT { return data_.rend(); }
      const_reverse_iterator crend() const BOOST_NOEXCEPT { return data_.crend(); }


      // Capacity
      bool empty() const BOOST_NOEXCEPT { return data_.empty(); }
      size_type size() const BOOST_NOEXCEPT { return data_.size(); }
      size_type max_size() const BOOST_NOEXCEPT { return data_.max_size(); } 
      void reserve(size_type const new_capacity) { data_.reserve(new_capacity); }
      size_type capacity() const BOOST_NOEXCEPT { return data_.capacity(); }
      void shrink_to_fit() { data_.shrink_to_fit(); }


      // Modifiers
      void swap(state& other)
        BOOST_NOEXCEPT_IF(( KET_is_nothrow_swappable<data_type>::value ))
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
        data_type result(::ket::utility::integer_exp2<std::size_t>(num_local_qubits), value_type(0));

        using ::ket::mpi::permutate_bits;
        std::pair<yampi::rank, StateInteger> const rank_index
          = ::ket::mpi::utility::qubit_value_to_rank_index(
              mpi_policy, result, permutate_bits(permutation, initial_integer));

        if (communicator.rank(environment) == rank_index.first)
          result[rank_index.second] = value_type(1);

        return result;
      }
    };


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
      BOOST_NOEXCEPT_IF((
        KET_is_nothrow_swappable<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >::value ))
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
      };


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
      };


      template <>
      struct for_each_local_range<0>
      {
        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, 0, Allocator>& call(
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          typedef std::vector<Complex, Allocator> dummy_local_state_type;
          typedef
            ::ket::mpi::utility::dispatch::for_each_local_range<
              ::ket::mpi::utility::policy::general_mpi, dummy_local_state_type>
            for_each_local_range_type;
          for_each_local_range_type::call(
            local_state.data(), KET_FORWARD_OR_COPY(Function, function));
          return local_state;
        }

        template <
          typename Complex, typename Allocator,
          typename Function>
        static ::ket::mpi::state<Complex, 0, Allocator> const& call(
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          typedef std::vector<Complex, Allocator> dummy_local_state_type;
          typedef
            ::ket::mpi::utility::dispatch::for_each_local_range<
              ::ket::mpi::utility::policy::general_mpi, dummy_local_state_type>
            for_each_local_range_type;
          for_each_local_range_type::call(
            local_state.data(), KET_FORWARD_OR_COPY(Function, function));
          return local_state;
        }
      };


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
      };


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
      };


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
      };
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
        };


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
        };


        template <typename MpiPolicy, typename LocalState_>
        struct for_each_local_range;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct for_each_local_range<
          ::ket::mpi::utility::policy::general_mpi,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          template <typename Function>
          static ::ket::mpi::state<Complex, num_page_qubits, Allocator>& call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            typedef
              ::ket::mpi::state_detail::for_each_local_range<num_page_qubits>
              for_each_local_range_type;
            return for_each_local_range_type::call(
              local_state, KET_FORWARD_OR_COPY(Function, function));
          }

          template <typename Function>
          static ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            typedef
              ::ket::mpi::state_detail::for_each_local_range<num_page_qubits>
              for_each_local_range_type;
            return for_each_local_range_type::call(
              local_state, KET_FORWARD_OR_COPY(Function, function));
          }
        };


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
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>
              transform_inclusive_scan_type;
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
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>
              transform_inclusive_scan_type;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation,
              initial_value, environment);
          }
        };


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
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>
              transform_inclusive_scan_self_type;
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
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>
              transform_inclusive_scan_self_type;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, initial_value, environment);
          }
        };


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
            typedef
              ::ket::mpi::state_detail::upper_bound<num_page_qubits>
              upper_bound_type;
            return upper_bound_type::call(local_state, value, compare, environment);
          }
        };
      } // namespace dispatch
    } // namespace utility
  } // namespace mpi
} // namespace ket


# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef KET_is_nothrow_swappable
# undef KET_true_type
# undef KET_array
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef KET_addressof

#endif

