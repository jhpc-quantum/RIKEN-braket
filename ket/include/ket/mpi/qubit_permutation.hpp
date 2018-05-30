#ifndef KET_MPI_QUBIT_PERMUTATION_HPP
# define KET_MPI_QUBIT_PERMUTATION_HPP

# include <boost/config.hpp>

# include <cassert>
# include <vector>
# include <memory>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
#   include <initializer_list>
# endif
# ifndef NDEBUG
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/is_unsigned.hpp>
#   endif
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/cstdint.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/algorithm/sort.hpp>
# include <boost/range/algorithm_ext/iota.hpp>

# include <ket/qubit.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
# else
#   define KET_is_unsigned boost::is_unsigned
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace mpi
  {
    // main interfaces:
    //   permutation[unpermutated_qubit]: get permutated qubit corresponding to unpermutated qubit unpermutated_qubit
    //   inverse(permutation): get inverse view of permutation
    //   inverse(permutation)[permutated_qubit]: get unpermutated qubit corresponding to permutated qubit permutated_qubit
    //   permutate(permutation, from1, from2): permutate qubits from1 and from2
    //   permutate_bits(permutation, unsigned_integer): convert unpermutated value unsigned_integer to permutated value
    //   inverse_permutate_bits(permutation, unsigned_integer): convert permutated value unsigned_integer to unpermutated value
    template <
      typename StateInteger = boost::uint64_t,
      typename BitInteger = unsigned int,
      typename Allocator
        = std::allocator< ::ket::qubit<StateInteger, BitInteger> > >
    class qubit_permutation
    {
      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      typedef
        std::vector<qubit_type, typename Allocator::template rebind<qubit_type>::other>
        data_type;
      typedef data_type inverse_data_type;

      data_type data_;
      inverse_data_type inverse_data_;

     public:
      typedef typename data_type::value_type value_type;
      typedef typename data_type::const_reference reference;
      typedef typename data_type::const_reference const_reference;
      typedef typename data_type::const_pointer pointer;
      typedef typename data_type::const_pointer const_pointer;
      typedef typename data_type::const_iterator iterator;
      typedef typename data_type::const_iterator const_iterator;
      typedef typename data_type::const_reverse_iterator reverse_iterator;
      typedef
        typename data_type::const_reverse_iterator const_reverse_iterator;
      typedef typename data_type::size_type size_type;
      typedef typename data_type::difference_type difference_type;
      typedef typename data_type::allocator_type allocator_type;

      qubit_permutation() BOOST_NOEXCEPT_OR_NOTHROW
        : data_(Allocator()), inverse_data_(Allocator())
      { }

      explicit qubit_permutation(Allocator const& allocator) BOOST_NOEXCEPT_OR_NOTHROW
        : data_(allocator), inverse_data_(allocator)
      { }

      explicit qubit_permutation(size_type const num_qubits)
        : data_(generate_identity_permutation(num_qubits)),
          inverse_data_(data_to_inverse_data(data_))
      { }

      template <typename InputIterator>
      qubit_permutation(
        InputIterator const first, InputIterator const last,
        Allocator const& allocator = Allocator())
        : data_(first, last, allocator),
          inverse_data_(data_to_inverse_data(data_, allocator))
      { assert(is_valid_permutation(data_)); }

      qubit_permutation(
        qubit_permutation const& other, Allocator const& allocator)
        : data_(other.data_, allocator),
          inverse_data_(other.inverse_data_, allocator)
      { }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      qubit_permutation(qubit_permutation&& other, Allocator const& allocator)
        : data_(std::move(other.data_), allocator),
          inverse_data_(std::move(other.inverse_data_), allocator)
      { }
# endif

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
      qubit_permutation(
        std::initializer_list<value_type> initializer_list,
        Allocator const& allocator = Allocator{})
        : data_(initializer_list, allocator),
          inverse_data_(data_to_inverse_data_(data_, allocator))
      { assert(is_valid_permutation(data_)); }

      qubit_permutation& operator=(
        std::initializer_list<value_type> initializer_list)
      {
        data_ = initializer_list;
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
        return *this;
      }
# endif

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
      ~qubit_permutation() = default;
      qubit_permutation(qubit_permutation const&) = default;
      qubit_permutation& operator=(qubit_permutation const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
      qubit_permutation(qubit_permutation&&) = default;
      qubit_permutation& operator=(qubit_permutation&&) = default;
#   endif
# endif

      template <typename InputIterator>
      void assign(InputIterator const first, InputIterator const last)
      {
        data_.assign(first, last);
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
      }

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
      void assign(std::initializer_list<value_type> initializer_list)
      {
        data_ = initializer_list;
        assert(is_valid_permutation(data_));
        inverse_data_ = data_to_inverse_data(data_);
      }
# endif

      allocator_type get_allocator() const BOOST_NOEXCEPT_OR_NOTHROW
      { return data_.get_allocator(); }

      const_iterator begin() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.begin(); }
      const_iterator end() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.end(); }
      const_reverse_iterator rbegin() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.rbegin(); }
      const_reverse_iterator rend() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.rend(); }

//        const_iterator cbegin() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.cbegin(); }
//        const_iterator cend() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.cend(); }
//        const_reverse_iterator crbegin() const BOOST_NOEXCEPT_OR_NOTHROW
//        { return data_.crbegin(); }
//        const_reverse_iterator crend() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.crend(); }

      size_type size() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.size(); }
      size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.max_size(); }
      size_type capacity() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.capacity(); }
      bool empty() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.empty(); }
      void reserve(size_type const size)
      { data_.reserve(size); inverse_data_.reserve(size); }
//        void shrink_to_fit()
//        { data_.shrink_to_fit(); inverse_data_.shrink_to_fit(); }

      const_reference operator[](value_type const from) const
      { return data_[static_cast<BitInteger>(from)]; }

      const_reference at(value_type const from) const
      { return data_.at(static_cast<BitInteger>(from)); }

      const_reference front() const { return data_.front(); }
      const_reference back() const { return data_.back(); }

//        value_type* data() BOOST_NOEXCEPT_OR_NOTHROW { return data_.data(); }
//        value_type const* data() const BOOST_NOEXCEPT_OR_NOTHROW { return data_.data(); }

      class inverse_view
      {
        inverse_data_type& inverse_data_;

       public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
        inverse_view() = delete;
        inverse_view& operator=(inverse_view const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        inverse_view& operator=(inverse_view&&) = delete;
#   endif
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
       private:
        inverse_view();
        inverse_view& operator=(inverse_view const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        inverse_view& operator=(inverse_view&&);
#   endif

       public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
        inverse_view(inverse_view const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        inverse_view(inverse_view&&) = default;
#   endif
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
        inverse_view(inverse_view const& other)
          : inverse_data_(other.inverse_data_)
        { }

#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        inverse_view(inverse_view&& other)
          : inverse_data_(std::move(other.inverse_data_))
        { }
#   endif
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

        inverse_view(inverse_data_type& inverse_data)
          : inverse_data_(inverse_data)
        { }

        const_iterator begin() const BOOST_NOEXCEPT_OR_NOTHROW { return inverse_data_.begin(); }
        const_iterator end() const BOOST_NOEXCEPT_OR_NOTHROW { return inverse_data_.end(); }
        const_reverse_iterator rbegin() const BOOST_NOEXCEPT_OR_NOTHROW
        { return inverse_data_.rbegin(); }
        const_reverse_iterator rend() const BOOST_NOEXCEPT_OR_NOTHROW
        { return inverse_data_.rend(); }

//          const_iterator cbegin() const BOOST_NOEXCEPT_OR_NOTHROW
//          { return inverse_data_.cbegin(); }
//          const_iterator cend() const BOOST_NOEXCEPT_OR_NOTHROW { return inverse_data_.cend(); }
//          const_reverse_iterator crbegin() const BOOST_NOEXCEPT_OR_NOTHROW
//          { return inverse_data_.crbegin(); }
//          const_reverse_iterator crend() const BOOST_NOEXCEPT_OR_NOTHROW
//          { return inverse_data_.crend(); }

        const_reference operator[](value_type const to) const
        { return inverse_data_[static_cast<BitInteger>(to)]; }

        const_reference at(value_type const to) const
        { return inverse_data_.at(static_cast<BitInteger>(to)); }

        const_reference front() const { return inverse_data_.front(); }
        const_reference back() const { return inverse_data_.back(); }

//          value_type* data() BOOST_NOEXCEPT_OR_NOTHROW { return inverse_data_.data(); }
//          value_type const* data() const BOOST_NOEXCEPT_OR_NOTHROW { return inverse_data_.data(); }
      };

      inverse_view inverse() { return inverse_view(inverse_data_); }
      inverse_view inverse() const
      { return inverse_view(const_cast<inverse_data_type&>(inverse_data_)); }

      void swap(qubit_permutation& other)
        BOOST_NOEXCEPT_IF((
          ::ket::utility::is_nothrow_swappable<data_type>::value
          and ::ket::utility::is_nothrow_swappable<inverse_data_type>::value ))
      { data_.swap(other.data_); inverse_data_.swap(other.inverse_data_); }

      void push_back()
      {
        data_.emplace_back(data_.size());
        inverse_data_.emplace_back(inverse_data_.size());
      }

      void clear() BOOST_NOEXCEPT_OR_NOTHROW { data_.clear(); inverse_data_.clear(); }

      void permutate(value_type const from1, value_type const from2)
      {
        assert(from1 != from2);
        using std::swap;
        swap(
          inverse_data_[static_cast<BitInteger>(data_[static_cast<BitInteger>(from1)])],
          inverse_data_[static_cast<BitInteger>(data_[static_cast<BitInteger>(from2)])]);
        swap(
          data_[static_cast<BitInteger>(from1)],
          data_[static_cast<BitInteger>(from2)]);
      }

     private:
      data_type generate_identity_permutation(size_type const size)
      {
        data_type result(size);
        boost::iota(result, static_cast<value_type>(0u));
        return result;
      }

      inverse_data_type data_to_inverse_data(
        data_type const& data, Allocator const& allocator = Allocator())
      {
        inverse_data_type result(data.begin(), data.end(), allocator);

        size_type data_size = data.size();
        for (size_type index = 0u; index < data_size; ++index)
          result[static_cast<typename ::ket::meta::bit_integer_of<value_type>::type>(data[index])]
            = static_cast<value_type>(index);

        return result;
      }

      bool is_valid_permutation(data_type permutation) const
      {
        boost::sort(permutation);

        value_type previous_qubit = permutation.front();
        const_iterator const last = boost::end(permutation);

        for (const_iterator iter = ++boost::begin(permutation); iter != last; ++iter)
          if (*iter == previous_qubit)
            return false;
          else
            previous_qubit = *iter;

        return true;
      }
    };


    template <typename StateInteger, typename BitInteger, typename Allocator>
    inline void swap(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& lhs,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& rhs)
      BOOST_NOEXCEPT_IF((
        ::ket::utility::is_nothrow_swappable<
           ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> >::value ))
    { lhs.swap(rhs); }


    template <
      typename StateInteger, typename BitInteger, typename Allocator>
    inline typename ::ket::mpi::qubit_permutation<
      StateInteger, BitInteger, Allocator>::inverse_view
    inverse(::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation)
    { return permutation.inverse(); }


    template <
      typename StateInteger, typename BitInteger, typename Allocator>
    inline void permutate(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      ::ket::qubit<StateInteger, BitInteger> const from1,
      ::ket::qubit<StateInteger, BitInteger> const from2)
    { permutation.permutate(from1, from2); }


    namespace permutate_bits_detail
    {
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
      enum class boolean { true_ = true, false_ = false };

#   define BOOLEAN_TYPE ::ket::mpi::permutate_bits_detail::boolean
#   define BOOLEAN_VALUE(value) ::ket::mpi::permutate_bits_detail::boolean::value
# else
      namespace boolean_ { enum boolean { true_ = true, false_ = false }; }

#   define BOOLEAN_TYPE ::ket::mpi::permutate_bits_detail::boolean_::boolean
#   define BOOLEAN_VALUE(value) ::ket::mpi::permutate_bits_detail::boolean_::value
# endif

      inline bool to_bool(BOOLEAN_TYPE const b)
      { return static_cast<bool>(b); }


      struct permutate_bits_impl
      {
        template <
          typename StateInteger, typename BitInteger,
          typename Allocator, typename UnsignedInteger>
        static UnsignedInteger call(
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&
              permutation,
          UnsignedInteger unsigned_integer)
        {
          static_assert(
            KET_is_unsigned<UnsignedInteger>::value,
            "UnsignedInteger should be unsigned");

          typedef std::vector< BOOLEAN_TYPE > is_permutated_type;
          static is_permutated_type is_permutated;
          is_permutated.assign(permutation.size(), BOOLEAN_VALUE(false_));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          BOOST_CONSTEXPR_OR_CONST qubit_type first_bit(0u);
          qubit_type const last_bit(permutation.size());

          for (qubit_type bit = first_bit; bit < last_bit; ++bit)
          {
            if (::ket::mpi::permutate_bits_detail::to_bool(
                  is_permutated[static_cast<BitInteger>(bit)]))
              continue;

            qubit_type present_bit = bit;
            // 00000b00000
            UnsignedInteger present_bit_value
              = unsigned_integer
                bitand (static_cast<UnsignedInteger>(1u) << bit);

            do
            {
              qubit_type const previous_bit = present_bit;
              // 00000b00000
              UnsignedInteger const previous_bit_value = present_bit_value;
              present_bit = permutation[previous_bit];
              // 00a00000000
              present_bit_value
                = unsigned_integer
                  bitand (static_cast<UnsignedInteger>(1u) << present_bit);

              // xxbxxbxxxxx
              unsigned_integer
                  // xx0xxbxxxxx
                = (unsigned_integer
                   bitand
                   compl (static_cast<UnsignedInteger>(1u) << present_bit))
                  bitor
                  // 00b00000000
                  ((previous_bit_value >> previous_bit) << present_bit);

              is_permutated[static_cast<BitInteger>(present_bit)]
                = BOOLEAN_VALUE(true_);
            }
            while (present_bit != bit);
          }

          return unsigned_integer;
        }
      };


      struct inverse_permutate_bits_impl
      {
        template <
          typename StateInteger, typename BitInteger,
          typename Allocator, typename UnsignedInteger>
        static UnsignedInteger call(
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&
              permutation,
          UnsignedInteger unsigned_integer)
        {
          static_assert(
            KET_is_unsigned<UnsignedInteger>::value,
            "UnsignedInteger should be unsigned");

          typedef std::vector< BOOLEAN_TYPE > is_permutated_type;
          static is_permutated_type is_permutated;
          is_permutated.assign(permutation.size(), BOOLEAN_VALUE(false_));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          BOOST_CONSTEXPR_OR_CONST qubit_type first_bit(0u);
          qubit_type const last_bit(permutation.size());

          for (qubit_type bit = first_bit; bit < last_bit; ++bit)
          {
            if (::ket::mpi::permutate_bits_detail::to_bool(
                  is_permutated[static_cast<BitInteger>(bit)]))
              continue;

            qubit_type present_bit = bit;
            // 00000b00000
            UnsignedInteger present_bit_value
              = unsigned_integer
                bitand (static_cast<UnsignedInteger>(1u) << bit);

            do
            {
              qubit_type const previous_bit = present_bit;
              // 00000b00000
              UnsignedInteger const previous_bit_value = present_bit_value;
              present_bit
                = ::ket::mpi::inverse(permutation)[previous_bit];
              // 00a00000000
              present_bit_value
                = unsigned_integer
                  bitand (static_cast<UnsignedInteger>(1u) << present_bit);

              // xxbxxbxxxxx
              unsigned_integer
                  // xx0xxbxxxxx
                = (unsigned_integer
                   bitand
                   compl (static_cast<UnsignedInteger>(1u) << present_bit))
                  bitor
                  // 00b00000000
                  ((previous_bit_value >> previous_bit) << present_bit);

              is_permutated[static_cast<BitInteger>(present_bit)]
                = BOOLEAN_VALUE(true_);
            }
            while (present_bit != bit);
          }

          return unsigned_integer;
        }
      };

# undef BOOLEAN_TYPE
# undef BOOLEAN_VALUE
    }


    template <
      typename StateInteger, typename BitInteger,
      typename Allocator, typename UnsignedInteger>
    inline UnsignedInteger permutate_bits(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
        permutation,
      UnsignedInteger const unsigned_integer)
    {
      return ::ket::mpi::permutate_bits_detail::permutate_bits_impl::call(
        permutation, unsigned_integer);
    }

    template <
      typename StateInteger, typename BitInteger,
      typename Allocator, typename UnsignedInteger>
    inline UnsignedInteger inverse_permutate_bits(
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const&
        permutation,
      UnsignedInteger const unsigned_integer)
    {
      return ::ket::mpi::permutate_bits_detail::inverse_permutate_bits_impl::call(
        permutation, unsigned_integer);
    }
  }
}


# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

