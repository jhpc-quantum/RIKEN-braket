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
# else
#   include <boost/type_traits/integral_constant.hpp>
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

/*
# include <boost/utility.hpp> // boost::prior
*/

# include <boost/iterator/iterator_facade.hpp>

# include <boost/range/sub_range.hpp>
# include <boost/range/iterator_range.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/algorithm/find_if.hpp>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/is_nothrow_swappable.hpp>
# include <ket/utility/loop_n.hpp>
//# include <ket/utility/parallel/loop_n.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
/*
# include <ket/mpi/utility/transform_inclusive_scan.hpp>
# include <ket/mpi/utility/transform_inclusive_scan_self.hpp>
# include <ket/mpi/utility/upper_bound.hpp>
*/
# include <ket/mpi/utility/detail/swap_local_qubits.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_true_type std::true_type
# else
#   define KET_true_type boost::true_type
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
# ifndef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      typedef boost::sub_range<data_type> page_range_type;
# else
      typedef boost::iterator_range<typename data_type::pointer> page_range_type;
# endif

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
          boost::begin(page_ranges_[page_nonpage_index_pair1.first])[page_nonpage_index_pair1.second],
          boost::begin(page_ranges_[page_nonpage_index_pair2.first])[page_nonpage_index_pair2.second]);
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
          (boost::begin(page_ranges_[get_page_id(index)])-data_.begin())+get_index_in_page(index));
      }

      const_reference at(size_type const index) const
      {
        return data_.at(
          (boost::begin(page_ranges_[get_page_id(index)])-data_.begin())+get_index_in_page(index));
      }

      reference operator[](size_type const index)
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));
        return boost::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      const_reference operator[](size_type const index) const
      {
        assert(index < (static_cast<size_type>(1u) << num_local_qubits_));
        return boost::begin(page_ranges_[get_page_id(index)])[get_index_in_page(index)];
      }

      reference front() { return *boost::begin(page_ranges_[get_page_id(0u)]); }
      const_reference front() const { return *boost::begin(page_ranges_[get_page_id(0u)]); }

      reference back() { return *--boost::end(page_ranges_[get_page_id((1u << num_local_qubits_)-1u)]); }
      const_reference back() const { return *--boost::end(page_ranges_[get_page_id((1u << num_local_qubits_)-1u)]); }


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
          ::ket::utility::is_nothrow_swappable<data_type>::value
          and ::ket::utility::is_nothrow_swappable<std::size_t>::value
          and ::ket::utility::is_nothrow_swappable< KET_array<page_range_type, num_pages> >::value
          and ::ket::utility::is_nothrow_swappable<page_range_type>::value ))
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
# ifndef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
          result[page_id]
            = boost::make_iterator_range(
                data.begin()+page_id*page_size, data.begin()+(page_id+1u)*page_size);
# else // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
          result[page_id]
            = boost::make_iterator_range(
                KET_addressof(data.front())+page_id*page_size,
                KET_addressof(data.front())+(page_id+1u)*page_size);
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
        }

        return result;
      }

      page_range_type generate_initial_buffer_range(data_type& data) const
      {
        assert(data.size() % (num_pages+1u) == 0u);
        size_type const page_size = data.size() / (num_pages+1u);

# ifndef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
        return boost::make_iterator_range(
          data.begin()+num_pages*page_size, data.begin()+(num_pages+1u)*page_size);
# else // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
        return boost::make_iterator_range(
          KET_addressof(data.front())+num_pages*page_size,
          KET_addressof(data.front())+(num_pages+1u)*page_size);
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
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
        template <
          typename Allocator, typename Complex, typename Allocator_, typename StateInteger>
        static void call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          std::vector<Complex, Allocator_>&,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype const datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          static_assert(
            num_page_qubits >= 1,
            "num_page_qubits should be at least 1 if using this function");
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
              typename boost::range_iterator<page_range_type>::type
              page_iterator;
            page_iterator const page_first = boost::begin(page_range);
            page_iterator const page_last = boost::end(page_range);
            page_iterator const buffer_first = boost::begin(local_state.buffer_range());

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

            yampi::algorithm::swap(
              yampi::ignore_status(), communicator, environment,
              yampi::make_buffer(the_first, the_last, datatype),
              yampi::make_buffer(
                the_buffer_first, the_buffer_last, datatype),
              target_rank);

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
              boost::begin(local_state.page_range(page_id)),
              boost::end(local_state.page_range(page_id)));
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
              boost::begin(local_state.page_range(page_id)),
              boost::end(local_state.page_range(page_id)));
          return local_state;
        }
      };


      /*
      template <int num_page_qubits>
      struct transform_inclusive_scan
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          impl(
            typename std::iterator_traits<ForwardIterator>::iterator_category(),
            parallel_policy, local_state, d_first, binary_operation, unary_operation);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          impl(
            typename std::iterator_traits<ForwardIterator>::iterator_category(),
            parallel_policy, local_state, d_first, binary_operation, unary_operation, initial_value);
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
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          Complex partial_sum = static_cast<Complex>(0);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ForwardIterator prev_d_first = d_first;
            d_first
              = ::ket::utility::transform_inclusive_scan(
                  parallel_policy,
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  d_first, binary_operation, unary_operation, partial_sum);
            std::advance(prev_d_first, boost::size(local_state.page_range(page_id))-1);
            partial_sum = *prev_d_first;
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex impl(
          std::forward_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ForwardIterator prev_d_first = d_first;
            d_first
              = ::ket::utility::transform_inclusive_scan(
                  parallel_policy,
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  d_first, binary_operation, unary_operation, partial_sum);
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
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          Complex partial_sum = static_cast<Complex>(0);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::transform_inclusive_scan(
                  parallel_policy,
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  d_first, binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(d_first);
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex impl(
          std::bidirectional_iterator_tag const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            d_first
              = ::ket::utility::transform_inclusive_scan(
                  parallel_policy,
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  d_first, binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(d_first);
          }

          return partial_sum;
        }
      };


      template <int num_page_qubits>
      struct transform_inclusive_scan_self
      {
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          Complex partial_sum = static_cast<Complex>(0);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ::ket::utility::transform_inclusive_scan(
              parallel_policy,
              boost::begin(local_state.page_range(page_id)),
              boost::end(local_state.page_range(page_id)),
              boost::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(boost::end(local_state.page_range(page_id)));
          }

          return partial_sum;
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          Complex partial_sum = static_cast<Complex>(initial_value);

          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            ::ket::utility::transform_inclusive_scan(
              parallel_policy,
              boost::begin(local_state.page_range(page_id)),
              boost::end(local_state.page_range(page_id)),
              boost::begin(local_state.page_range(page_id)),
              binary_operation, unary_operation, partial_sum);
            partial_sum = *boost::prior(boost::end(local_state.page_range(page_id)));
          }

          return partial_sum;
        }
      };


      // Usually num_page_qubits is small, so linear search for page is probably not bad.
      template <int num_page_qubits>
      struct upper_bound
      {
        template <typename Complex, typename Allocator>
        static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          Complex const& value)
        {
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          typedef typename local_state_type::difference_type difference_type;

          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            std::size_t index_in_page
              = std::upper_bound(
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  value)
                - boost::begin(local_state.page_range(page_id));

            if (index_in_page < boost::size(local_state.page_range(page_id)))
            {
              std::size_t const num_qubits_in_page
                = local_state.num_local_qubits()-num_page_qubits;
              return static_cast<difference_type>((page_id << num_qubits_in_page) bitor index_in_page);
            }
          }

          return static_cast<difference_type>(local_state.size());
        }

        template <typename Complex, typename Allocator, typename Compare>
        static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
          Complex const& value, Compare compare)
        {
          typedef ::ket::mpi::state<Complex, num_page_qubits, Allocator> local_state_type;
          typedef typename local_state_type::difference_type difference_type;

          for (std::size_t page_id = 0u; page_id < local_state_type::num_pages; ++page_id)
          {
            std::size_t index_in_page
              = std::upper_bound(
                  boost::begin(local_state.page_range(page_id)),
                  boost::end(local_state.page_range(page_id)),
                  value, compare)
                - boost::begin(local_state.page_range(page_id));

            if (index_in_page < boost::size(local_state.page_range(page_id)))
            {
              std::size_t const num_qubits_in_page
                = local_state.num_local_qubits()-num_page_qubits;
              return static_cast<difference_type>((page_id << num_qubits_in_page) bitor index_in_page);
            }
          }

          return static_cast<difference_type>(local_state.size());
        }
      };
      */
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
        BOOST_NOEXCEPT_IF(( ::ket::utility::is_nothrow_swappable<data_type>::value ))
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
        ::ket::utility::is_nothrow_swappable<
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
          yampi::datatype const datatype, yampi::rank const target_rank,
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


      /*
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
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          return ::ket::mpi::utility::transform_inclusive_scan(
            parallel_policy,
            local_state.data(), d_first, binary_operation, unary_operation);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ForwardIterator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          ForwardIterator const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          return ::ket::mpi::utility::transform_inclusive_scan(
            parallel_policy,
            local_state.data(), d_first, binary_operation, unary_operation, initial_value);
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
          BinaryOperation binary_operation, UnaryOperation unary_operation)
        {
          return ::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy,
            local_state.data(), binary_operation, unary_operation);
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static Complex call(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value)
        {
          return ::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy,
            local_state.data(), binary_operation, unary_operation, initial_value);
        }
      };


      template <>
      struct upper_bound<0>
      {
        template <typename Complex, typename Allocator>
        static typename ::ket::mpi::state<Complex, 0, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          Complex const& value)
        { return ::ket::mpi::utility::upper_bound(local_state.data(), value); }

        template <typename Complex, typename Allocator, typename Compare>
        static typename ::ket::mpi::state<Complex, 0, Allocator>::difference_type call(
          ::ket::mpi::state<Complex, 0, Allocator> const& local_state,
          Complex const& value, Compare compare)
        { return ::ket::mpi::utility::upper_bound(local_state.data(), value, compare); }
      };
      */
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
            yampi::datatype const datatype, yampi::rank const target_rank,
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


        /*
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
            BinaryOperation binary_operation, UnaryOperation unary_operation)
          {
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>
              transform_inclusive_scan_type;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation);
          }

          template <
            typename ParallelPolicy, typename ForwardIterator,
            typename BinaryOperation, typename UnaryOperation, typename Value>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            ForwardIterator const d_first,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value)
          {
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan<num_page_qubits>
              transform_inclusive_scan_type;
            return transform_inclusive_scan_type::call(
              parallel_policy,
              local_state, d_first, binary_operation, unary_operation, initial_value);
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
            BinaryOperation binary_operation, UnaryOperation unary_operation)
          {
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>
              transform_inclusive_scan_self_type;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation);
          }

          template <
            typename ParallelPolicy,
            typename BinaryOperation, typename UnaryOperation, typename Value>
          static Complex call(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits, Allocator>& local_state,
            BinaryOperation binary_operation, UnaryOperation unary_operation,
            Value const initial_value)
          {
            typedef
              ::ket::mpi::state_detail::transform_inclusive_scan_self<num_page_qubits>
              transform_inclusive_scan_self_type;
            return transform_inclusive_scan_self_type::call(
              parallel_policy,
              local_state, binary_operation, unary_operation, initial_value);
          }
        };


        template <typename LocalState_>
        struct upper_bound;

        template <typename Complex, int num_page_qubits, typename Allocator>
        struct upper_bound<
          ::ket::mpi::state<Complex, num_page_qubits, Allocator> >
        {
          static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            Complex const& value)
          {
            typedef
              ::ket::mpi::state_detail::upper_bound<num_page_qubits>
              upper_bound_type;
            return upper_bound_type::call(local_state, value);
          }

          template <typename Compare>
          static typename ::ket::mpi::state<Complex, num_page_qubits, Allocator>::difference_type call(
            ::ket::mpi::state<Complex, num_page_qubits, Allocator> const& local_state,
            Complex const& value, Compare compare)
          {
            typedef
              ::ket::mpi::state_detail::upper_bound<num_page_qubits>
              upper_bound_type;
            return upper_bound_type::call(local_state, value, compare);
          }
        };
        */
      } // namespace dispatch
    } // namespace utility
  } // namespace mpi
} // namespace ket


# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef KET_true_type
# undef KET_array
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef KET_addressof

#endif

