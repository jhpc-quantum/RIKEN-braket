#ifndef YAMPI_DERIVED_DATATYPE_HPP
# define YAMPI_DERIVED_DATATYPE_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <vector>
# include <algorithm>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/range/value_type.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/algorithm/transform.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/error.hpp>
# include <yampi/allocator.hpp>
# include <yampi/access.hpp>
# include <yampi/address.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class strided_block
  {
    int length_;
    int stride_;

   public:
    strided_block(int const length, int const stride)
      : length_(length), stride_(stride)
    { }

    int const& length() const { return length_; }
    int const& stride() const { return stride_; }

    void swap(strided_block& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(length_, other.length_);
      swap(stride_, other.stride_);
    }
  };

  inline bool operator==(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs)
  { return lhs.length() == rhs.length() and lhs.stride() == rhs.stride(); }

  inline bool operator!=(
    ::yampi::strided_block const& lhs, ::yampi::strided_block const& rhs)
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::strided_block& lhs, ::yampi::strided_block& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs.swap(rhs); }


  class displaced_block
  {
    int length_;
    int displacement_;

   public:
    displaced_block(int const length, int const displacement)
      : length_(length), displacement_(displacement)
    { }

    int const& length() const { return length_; }
    int const& displacement() const { return displacement_; }

    void swap(displaced_block& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(length_, other.length_);
      swap(displacement_, other.displacement_);
    }
  };

  inline bool operator==(
    ::yampi::displaced_block const& lhs, ::yampi::displaced_block const& rhs)
  { return lhs.length() == rhs.length() and lhs.displacement() == rhs.displacement(); }

  inline bool operator!=(
    ::yampi::displaced_block const& lhs, ::yampi::displaced_block const& rhs)
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::displaced_block& lhs, ::yampi::displaced_block& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs.swap(rhs); }


  template <typename Allocator = ::yampi::allocator<int> >
  class displaced_blocks
  {
   public:
    typedef typename Allocator::template rebind<int>::other allocator_type;
    typedef std::vector<int, allocator_type> block_lengths_type;
    typedef block_lengths_type displacements_type;

   private:
    block_lengths_type block_lengths_;
    displacements_type displacements_;

   public:
    typedef ::yampi::displaced_block value_type;
    typedef typename block_lengths_type::size_type size_type;
    typedef typename block_lengths_type::difference_type difference_type;

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    displaced_blocks() = default;

    displaced_blocks(displaced_blocks const&) = default;
    displaced_blocks& operator=(displaced_blocks const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    displaced_blocks(displaced_blocks&&) = default;
    displaced_blocks& operator=(displaced_blocks&&) = default;
#   endif
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    displaced_blocks(
      allocator_type const& block_length_allocator, allocator_type const& displacement_allocator)
      : block_lengths_(block_length_allocator), displacements_(displacement_allocator)
    { }

    explicit displaced_blocks(size_type const size)
      : block_lengths_(size), displacements_(size)
    { }

    displaced_blocks(
      size_type const size,
      allocator_type const& block_length_allocator, allocator_type const& displacement_allocator)
      : block_lengths_(size, int(), block_length_allocator),
        displacements_(size, int(), displacement_allocator)
    { }

    displaced_blocks(size_type const size, ::yampi::displaced_block const& block)
      : block_lengths_(size, block.length()), displacements_(size, block.displacement())
    { }

    displaced_blocks(
      size_type const size, ::yampi::displaced_block const& block,
      allocator_type const& block_length_allocator, allocator_type const& displacement_allocator)
      : block_lengths_(size, block.length(), block_length_allocator),
        displacements_(size, block.displacement(), displacement_allocator)
    { }

    template <typename InputIterator>
    displaced_blocks(InputIterator const first, InputIterator const last)
      : block_lengths_(), displacements_()
    { this->assign(first, last); }

    template <typename InputIterator>
    displaced_blocks(
      InputIterator const first, InputIterator const last,
      allocator_type const& block_length_allocator, allocator_type const& displacement_allocator)
      : block_lengths_(block_length_allocator), displacements_(displacement_allocator)
    { this->assign(first, last); }

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    displaced_blocks(std::initializer_list< ::yampi::displaced_block > initializer_list)
      : block_lengths_(), displacements_()
    { this->assign(initializer_list); }

    displaced_blocks(
      std::initializer_list< ::yampi::displaced_block > initializer_list,
      allocator_type const& block_length_allocator, allocator_type const& displacement_allocator)
      : block_lengths_(block_length_allocator), displacements_(displacement_allocator)
    { this->assign(initializer_list); }

    displaced_blocks& operator=(
      std::initializer_list< ::yampi::displaced_block > initializer_list)
    { this->assign(initializer_list); }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST


    block_lengths_type const& block_lengths() const { return block_lengths_; }
    displacements_type const& displacements() const { return displacements_; }


    void assign(size_type const size, ::yampi::displaced_block const& block)
    {
      block_lengths_.assign(size, block.length());
      displacements_.assign(size, block.displacement());
    }

    template <typename InputIterator>
    void assign(InputIterator const first, InputIterator const last)
    { do_assign(first, last, std::iterator_traits<InputIterator>::iterator_category()); }

   private:
    template <typename InputIterator>
    void do_assign(InputIterator first, InputIterator const last, std::input_iterator_tag)
    {
      for (; first != last; ++first)
      {
        block_lengths_.push_back(first->length());
        displacements_.push_back(first->displacement());
      }
    }

    template <typename InputIterator>
    void do_assign(InputIterator first, InputIterator const last, std::random_access_iterator_tag)
    {
      block_lengths_.reserve(last-first);
      displacements_.reserve(last-first);

      for (; first != last; ++first)
      {
        block_lengths_.push_back(first->length());
        displacements_.push_back(first->displacement());
      }
    }

   public:
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    void assign(std::initializer_list< ::yampi::displaced_block > initializer_list)
    {
      block_lengths_.clear();
      displacements_.clear();
      block_lengths_.reserve(initializer_list.size());
      displacements_.reserve(initializer_list.size());

#   ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
      for (::yampi::displaced_block const& block: initializer_list)
      {
        block_lengths_.push_back(block.length());
        displacements_.push_back(block.displacement());
      }
#   else // BOOST_NO_CXX11_RANGE_BASED_FOR
      typedef typename std::initializer_list::const_iterator iterator;
      for (iterator iter = initializer_list.begin(), last = initializer_list.end();
           iter != last; ++iter)
      {
        block_lengths_.push_back(iter->length());
        displacements_.push_back(iter->displacement());
      }
#   endif // BOOST_NO_CXX11_RANGE_BASED_FOR
    }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST

    bool empty() const BOOST_NOEXCEPT
    {
# ifdef NDEBUG
      return block_lengths_.empty();
# else
      return block_lengths_.empty() and displacements_.empty();
# endif
    }

    size_type size() const BOOST_NOEXCEPT
    {
      assert(block_lengths_.size() == displacements_.size());
      return block_lengths_.size();
    }

    void reserve(size_type const new_capacity)
    {
      block_lengths_.reserve(new_capacity);
      displacements_.reserve(new_capacity);
    }

    size_type capacity() const BOOST_NOEXCEPT
    {
      assert(block_lengths_.capacity() == displacements_.capacity());
      return block_lengths_.capacity();
    }

    void clear() BOOST_NOEXCEPT
    {
      block_lengths_.clear();
      displacements_.clear();
    }

    void push_back(::yampi::displaced_block const& block)
    {
      block_lengths_.push_back(block.length());
      displacements_.push_back(block.displacement());
    }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void push_back(::yampi::displaced_block&& block)
    {
      block_lengths_.push_back(std::move(block.length()));
      displacements_.push_back(std::move(block.displacement()));
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void pop_back()
    {
      block_lengths_.pop_back();
      displacements_.pop_back();
    }

    void resize(size_type const size)
    {
      block_lengths_.resize(size);
      displacements_.resize(size);
    }

    void resize(size_type const size, ::yampi::displaced_block const& block)
    {
      block_lengths_.resize(size, block.length());
      displacements_.resize(size, block.displacement());
    }

    void swap(displaced_blocks& other)
      BOOST_NOEXCEPT_IF((
        ::yampi::utility::is_nothrow_swappable<block_lengths_type>::value
        and ::yampi::utility::is_nothrow_swappable<displacements_type>::value ))
    {
      using std::swap;
      swap(block_lengths_, other.block_lengths_);
      swap(displacements_, other.displacements_);
    }
  };

  template <typename Allocator>
  inline bool operator==(
    ::yampi::displaced_blocks<Allocator> const& lhs,
    ::yampi::displaced_blocks<Allocator> const& rhs)
  {
    return lhs.block_lengths() == rhs.block_lengths()
      and lhs.displacements() == rhs.displacements();
  }

  template <typename Allocator>
  inline bool operator!=(
    ::yampi::displaced_blocks<Allocator> const& lhs,
    ::yampi::displaced_blocks<Allocator> const& rhs)
  { return not (lhs == rhs); }

  template <typename Allocator>
  inline void swap(
    ::yampi::displaced_blocks<Allocator>& lhs,
    ::yampi::displaced_blocks<Allocator>& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable< ::yampi::displaced_blocks<Allocator> >::value ))
  { lhs.swap(rhs); }


  template <typename Allocator = ::yampi::allocator<int> >
  class displaced_constant_blocks
  {
   public:
    typedef typename Allocator::template rebind<int>::other allocator_type;
    typedef std::vector<int, allocator_type> displacements_type;

   private:
    int block_length_;
    displacements_type displacements_;

   public:
    typedef ::yampi::displaced_block value_type;
    typedef typename displacements_type::size_type size_type;
    typedef typename displacements_type::difference_type difference_type;

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    displaced_constant_blocks() = default;

    displaced_constant_blocks(displaced_constant_blocks const&) = default;
    displaced_constant_blocks& operator=(displaced_constant_blocks const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    displaced_constant_blocks(displaced_constant_blocks&&) = default;
    displaced_constant_blocks& operator=(displaced_constant_blocks&&) = default;
#   endif
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    explicit displaced_constant_blocks(allocator_type const& allocator)
      : block_length_(), displacements_(allocator)
    { }

    explicit displaced_constant_blocks(size_type const size)
      : block_length_(), displacements_(size)
    { }

    displaced_constant_blocks(
      size_type const size, allocator_type const& allocator)
      : block_length_(), displacements_(size, int(), allocator)
    { }

    displaced_constant_blocks(
      size_type const size, int const block_length, int const displacement)
      : block_length_(block_length), displacements_(size, displacement)
    { }

    displaced_constant_blocks(
      size_type const size, int const block_length, int const displacement,
      allocator_type const& allocator)
      : block_length_(block_length), displacements_(size, displacement, allocator)
    { }

    template <typename InputIterator>
    displaced_constant_blocks(
      int const block_length, InputIterator const first, InputIterator const last)
      : block_length_(block_length), displacements_(first, last)
    { }

    template <typename InputIterator>
    displaced_constant_blocks(
      int const block_length, InputIterator const first, InputIterator const last,
      allocator_type const& allocator)
      : block_length_(block_length), displacements_(first, last, allocator)
    { }

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    displaced_constant_blocks(
      int const block_length, std::initializer_list<int> initializer_list)
      : block_length_(block_length), displacements_(initializer_list)
    { }

    displaced_constant_blocks(
      int const block_length, std::initializer_list<int> initializer_list,
      allocator_type const& allocator)
      : block_length_(block_length), displacements_(initializer_list, allocator)
    { }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST


    int block_length() const { return block_length_; }
    void block_length(int new_block_length) { block_length_ = new_block_length; }
    displacements_type const& displacements() const { return displacements_; }


    void assign(size_type const size, int const displacement)
    { displacements_.assign(size, displacement); }

    template <typename InputIterator>
    void assign(InputIterator const first, InputIterator const last)
    { displacements_.assign(first, last); }

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    void assign(std::initializer_list<int> initializer_list)
    { displacements_ = initializer_list; }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST

    bool empty() const BOOST_NOEXCEPT { return displacements_.empty(); }
    size_type size() const BOOST_NOEXCEPT { return displacements_.size(); }
    void reserve(size_type const new_capacity) { displacements_.reserve(new_capacity); }
    size_type capacity() const BOOST_NOEXCEPT { return displacements_.capacity(); }

    void clear() BOOST_NOEXCEPT
    { displacements_.clear(); }

    void push_back(int const displacement)
    { displacements_.push_back(displacement); }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void push_back(int&& displacement)
    { displacements_.push_back(std::move(displacement)); }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void pop_back()
    { displacements_.pop_back(); }

    void resize(size_type const size)
    { displacements_.resize(size); }

    void resize(size_type const size, int const displacement)
    { displacements_.resize(size, displacement); }

    void swap(displaced_constant_blocks& other)
      BOOST_NOEXCEPT_IF((
        ::yampi::utility::is_nothrow_swappable<int>::value
        and ::yampi::utility::is_nothrow_swappable<displacements_type>::value ))
    {
      using std::swap;
      swap(block_length_, other.block_length_);
      swap(displacements_, other.displacements_);
    }
  };

  template <typename Allocator>
  inline bool operator==(
    ::yampi::displaced_constant_blocks<Allocator> const& lhs,
    ::yampi::displaced_constant_blocks<Allocator> const& rhs)
  {
    return lhs.block_length() == rhs.block_length()
      and lhs.displacements() == rhs.displacements();
  }

  template <typename Allocator>
  inline bool operator!=(
    ::yampi::displaced_constant_blocks<Allocator> const& lhs,
    ::yampi::displaced_constant_blocks<Allocator> const& rhs)
  { return not (lhs == rhs); }

  template <typename Allocator>
  inline void swap(
    ::yampi::displaced_constant_blocks<Allocator>& lhs,
    ::yampi::displaced_constant_blocks<Allocator>& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable<
        ::yampi::displaced_constant_blocks<Allocator> >::value ))
  { lhs.swap(rhs); }


  class displaced_typed_block
  {
    ::yampi::datatype datatype_;
    int length_;
    MPI_Aint displacement_address_;

   public:
    template <typename Value, typename Base>
    displaced_typed_block(
      ::yampi::datatype const& datatype, int const length,
      Value const& value, Base const& base, ::yampi::environment const& environment)
      : datatype_(datatype),
        length_(length),
        displacement_address_(displacement_address(value, base, environment))
    { }

    template <typename Value>
    displaced_typed_block(
      ::yampi::datatype const& datatype, int const length,
      Value const& value, MPI_Aint const base_address, ::yampi::environment const& environment)
      : datatype_(datatype),
        length_(length),
        displacement_address_(displacement_address(value, base_address, environment))
    { }

   private:
    template <typename Value>
    MPI_Aint displacement_address(
      Value const& value, MPI_Aint const base_address, ::yampi::environment const& environment)
    { return ::yampi::address(value, environment) - base_address; }

    template <typename Value, typename Base>
    MPI_Aint displacement_address(
      Value const& value, Base const& base, ::yampi::environment const& environment)
    { return displacement_address(value, ::yampi::address(base, environment), environment); }

   public:
    ::yampi::datatype const& datatype() const { return datatype_; }
    int const& length() const { return length_; }
    MPI_Aint const& displacement_address() const { return displacement_address_; }

    void swap(displaced_typed_block& other)
      BOOST_NOEXCEPT_IF((
        ::yampi::utility::is_nothrow_swappable< ::yampi::datatype >::value
        and ::yampi::utility::is_nothrow_swappable<int>::value
        and ::yampi::utility::is_nothrow_swappable<MPI_Aint>::value ))
    {
      using std::swap;
      swap(datatype_, other.datatype_);
      swap(length_, other.length_);
      swap(displacement_address_, other.displacement_address_);
    }
  };

  inline bool operator==(
    ::yampi::displaced_typed_block const& lhs, ::yampi::displaced_typed_block const& rhs)
  {
    return lhs.datatype() == rhs.datatype()
      and lhs.length() == rhs.length()
      and lhs.displacement_address() == rhs.displacement_address();
  }

  inline bool operator!=(
    ::yampi::displaced_typed_block const& lhs, ::yampi::displaced_typed_block const& rhs)
  { return not (lhs == rhs); }

  inline void swap(
    ::yampi::displaced_typed_block& lhs, ::yampi::displaced_typed_block& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable< ::yampi::displaced_block >::value ))
  { lhs.swap(rhs); }


  template <
    typename MpiDatatypeAllocator = ::yampi::allocator<MPI_Datatype>,
    typename BlockLengthAllocator = ::yampi::allocator<int>,
    typename MpiAddressAllocator = ::yampi::allocator<MPI_Aint> >
  class displaced_typed_blocks
  {
   public:
    typedef
      typename MpiDatatypeAllocator::template rebind<MPI_Datatype>::other
      mpi_datatype_allocator_type;
    typedef
      typename BlockLengthAllocator::template rebind<int>::other
      block_length_allocator_type;
    typedef
      typename MpiDatatypeAllocator::template rebind<MPI_Aint>::other
      mpi_address_allocator_type;
    typedef std::vector<MPI_Datatype, mpi_datatype_allocator_type> mpi_datatypes_type;
    typedef std::vector<int, block_length_allocator_type> block_lengths_type;
    typedef std::vector<MPI_Aint, mpi_address_allocator_type> displacement_addresses_type;

   private:
    mpi_datatypes_type mpi_datatypes_;
    block_lengths_type block_lengths_;
    displacement_addresses_type displacement_addresses_;

   public:
    typedef ::yampi::displaced_typed_block value_type;
    typedef typename block_lengths_type::size_type size_type;
    typedef typename block_lengths_type::difference_type difference_type;

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    displaced_typed_blocks() = default;

    displaced_typed_blocks(displaced_typed_blocks const&) = default;
    displaced_typed_blocks& operator=(displaced_typed_blocks const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    displaced_typed_blocks(displaced_typed_blocks&&) = default;
    displaced_typed_blocks& operator=(displaced_typed_blocks&&) = default;
#   endif
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    displaced_typed_blocks(
      mpi_datatype_allocator_type const& mpi_datatype_allocator,
      block_length_allocator_type const& block_length_allocator,
      mpi_address_allocator_type const& displacement_address_allocator)
      : mpi_datatypes_(mpi_datatype_allocator),
        block_lengths_(block_length_allocator),
        displacement_addresses_(displacement_address_allocator)
    { }

    explicit displaced_typed_blocks(size_type const size)
      : mpi_datatypes_(size), block_lengths_(size), displacement_addresses_(size)
    { }

    displaced_typed_blocks(
      size_type const size,
      mpi_datatype_allocator_type const& mpi_datatype_allocator,
      block_length_allocator_type const& block_length_allocator,
      mpi_address_allocator_type const& displacement_address_allocator)
      : mpi_datatypes_(size, MPI_Datatype(), mpi_datatype_allocator),
        block_lengths_(size, int(), block_length_allocator),
        displacement_addresses_(size, MPI_Aint(), displacement_address_allocator)
    { }

    displaced_typed_blocks(size_type const size, ::yampi::displaced_typed_block const& block)
      : mpi_datatypes_(size, block.datatype().mpi_datatype()),
        block_lengths_(size, block.length()),
        displacement_addresses_(size, block.displacement_address())
    { }

    displaced_typed_blocks(
      size_type const size, ::yampi::displaced_typed_block const& block,
      mpi_datatype_allocator_type const& mpi_datatype_allocator,
      block_length_allocator_type const& block_length_allocator,
      mpi_address_allocator_type const& displacement_address_allocator)
      : mpi_datatypes_(size, block.datatype().mpi_datatype(), mpi_datatype_allocator),
        block_lengths_(size, block.length(), block_length_allocator),
        displacement_addresses_(size, block.displacement_address(), displacement_address_allocator)
    { }

    template <typename InputIterator>
    displaced_typed_blocks(InputIterator const first, InputIterator const last)
      : mpi_datatypes_(), block_lengths_(), displacement_addresses_()
    { this->assign(first, last); }

    template <typename InputIterator>
    displaced_typed_blocks(
      InputIterator const first, InputIterator const last,
      mpi_datatype_allocator_type const& mpi_datatype_allocator,
      block_length_allocator_type const& block_length_allocator,
      mpi_address_allocator_type const& displacement_address_allocator)
      : mpi_datatypes_(mpi_datatype_allocator),
        block_lengths_(block_length_allocator),
        displacement_addresses_(displacement_address_allocator)
    { this->assign(first, last); }

# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    displaced_typed_blocks(
      std::initializer_list< ::yampi::displaced_typed_block > initializer_list)
      : mpi_datatypes_(), block_lengths_(), displacement_addresses_()
    { this->assign(initializer_list); }

    displaced_typed_blocks(
      std::initializer_list< ::yampi::displaced_typed_block > initializer_list,
      mpi_datatype_allocator_type const& mpi_datatype_allocator,
      block_length_allocator_type const& block_length_allocator,
      mpi_address_allocator_type const& displacement_address_allocator)
      : mpi_datatypes_(mpi_datatype_allocator),
        block_lengths_(block_length_allocator),
        displacement_addresses_(displacement_address_allocator)
    { this->assign(initializer_list); }

    displaced_typed_blocks& operator=(
      std::initializer_list< ::yampi::displaced_typed_block > initializer_list)
    { this->assign(initializer_list); }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST


    mpi_datatypes_type const& mpi_datatypes() const { return mpi_datatypes_; }
    block_lengths_type const& block_lengths() const { return block_lengths_; }
    displacement_addresses_type const& displacement_addresses() const
    { return displacement_addresses_; }


    void assign(size_type const size, ::yampi::displaced_typed_block const& block)
    {
      mpi_datatypes_.assign(block.datatype().mpi_datatype());
      block_lengths_.assign(size, block.length());
      displacement_addresses_.assign(size, block.displacement_address());
    }

    template <typename InputIterator>
    void assign(InputIterator const first, InputIterator const last)
    { do_assign(first, last, std::iterator_traits<InputIterator>::iterator_category()); }

   private:
    template <typename InputIterator>
    void do_assign(InputIterator first, InputIterator const last, std::input_iterator_tag)
    {
      for (; first != last; ++first)
      {
        mpi_datatypes_.push_back(first->datatype().mpi_datatype());
        block_lengths_.push_back(first->length());
        displacement_addresses_.push_back(first->displacement_address());
      }
    }

    template <typename InputIterator>
    void do_assign(InputIterator first, InputIterator const last, std::random_access_iterator_tag)
    {
      mpi_datatypes_.reserve(last-first);
      block_lengths_.reserve(last-first);
      displacement_addresses_.reserve(last-first);

      for (; first != last; ++first)
      {
        mpi_datatypes_.push_back(first->datatype().mpi_datatype());
        block_lengths_.push_back(first->length());
        displacement_addresses_.push_back(first->displacement_address());
      }
    }

   public:
# ifndef BOOST_NO_CXX11_HDR_INITIALIZER_LIST
    void assign(std::initializer_list< ::yampi::displaced_typed_block > initializer_list)
    {
      mpi_datatypes_.clear();
      block_lengths_.clear();
      displacement_addresses_.clear();
      mpi_datatypes_.reserve(initializer_list.size());
      block_lengths_.reserve(initializer_list.size());
      displacement_addresses_.reserve(initializer_list.size());

#   ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
      for (::yampi::displaced_typed_block const& block: initializer_list)
      {
        mpi_datatypes_.push_back(block.datatype().mpi_datatype());
        block_lengths_.push_back(block.length());
        displacement_addresses_.push_back(block.displacement_address());
      }
#   else // BOOST_NO_CXX11_RANGE_BASED_FOR
      typedef typename std::initializer_list::const_iterator iterator;
      for (iterator iter = initializer_list.begin(), last = initializer_list.end();
           iter != last; ++iter)
      {
        mpi_datatypes_.push_back(iter->datatype().mpi_datatype());
        block_lengths_.push_back(iter->length());
        displacement_addresses_.push_back(iter->displacement_address());
      }
#   endif // BOOST_NO_CXX11_RANGE_BASED_FOR
    }
# endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST

    bool empty() const BOOST_NOEXCEPT
    {
# ifdef NDEBUG
      return block_lengths_.empty();
# else
      return mpi_datatypes_.empty()
        and block_lengths_.empty()
        and displacement_addresses_.empty();
# endif
    }

    size_type size() const BOOST_NOEXCEPT
    {
      assert(
        block_lengths_.size() == displacement_addresses_.size()
        and block_lengths_.size() == mpi_datatypes_.size());
      return block_lengths_.size();
    }

    void reserve(size_type const new_capacity)
    {
      mpi_datatypes_.reserve(new_capacity);
      block_lengths_.reserve(new_capacity);
      displacement_addresses_.reserve(new_capacity);
    }

    size_type capacity() const BOOST_NOEXCEPT
    {
      assert(
        block_lengths_.capacity() == displacement_addresses_.capacity()
        and block_lengths_.capacity() == mpi_datatypes_.capacity());
      return block_lengths_.capacity();
    }

    void clear() BOOST_NOEXCEPT
    {
      mpi_datatypes_.clear();
      block_lengths_.clear();
      displacement_addresses_.clear();
    }

    void push_back(::yampi::displaced_typed_block const& block)
    {
      mpi_datatypes_.push_back(block.datatype().mpi_datatype());
      block_lengths_.push_back(block.length());
      displacement_addresses_.push_back(block.displacement_address());
    }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void push_back(::yampi::displaced_typed_block&& block)
    {
      mpi_datatypes_.push_back(std::move(block.datatype().mpi_datatype()));
      block_lengths_.push_back(std::move(block.length()));
      displacement_addresses_.push_back(std::move(block.displacement_address()));
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void pop_back()
    {
      mpi_datatypes_.pop_back();
      block_lengths_.pop_back();
      displacement_addresses_.pop_back();
    }

    void resize(size_type const size)
    {
      mpi_datatypes_.resize(size);
      block_lengths_.resize(size);
      displacement_addresses_.resize(size);
    }

    void resize(size_type const size, ::yampi::displaced_typed_block const& block)
    {
      mpi_datatypes_.resize(size, block.datatype().mpi_datatype());
      block_lengths_.resize(size, block.length());
      displacement_addresses_.resize(size, block.displacement_address());
    }

    void swap(displaced_typed_blocks& other)
      BOOST_NOEXCEPT_IF((
        ::yampi::utility::is_nothrow_swappable<mpi_datatypes_type>::value
        and ::yampi::utility::is_nothrow_swappable<block_lengths_type>::value
        and ::yampi::utility::is_nothrow_swappable<displacement_addresses_type>::value ))
    {
      using std::swap;
      swap(mpi_datatypes_, other.mpi_datatypes_);
      swap(block_lengths_, other.block_lengths_);
      swap(displacement_addresses_, other.displacement_addresses_);
    }
  };

  template <
    typename MpiDatatypeAllocator, typename BlockLengthAllocator, typename MpiAddressAllocator>
  inline bool operator==(
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& lhs,
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& rhs)
  {
    return lhs.mpi_datatypes() == rhs.mpi_datatypes()
      and lhs.block_lengths() == rhs.block_lengths()
      and lhs.displacement_addresses() == rhs.displacement_addresses();
  }

  template <
    typename MpiDatatypeAllocator, typename BlockLengthAllocator, typename MpiAddressAllocator>
  inline bool operator!=(
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& lhs,
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& rhs)
  { return not (lhs == rhs); }

  template <
    typename MpiDatatypeAllocator, typename BlockLengthAllocator, typename MpiAddressAllocator>
  inline void swap(
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator>& lhs,
    ::yampi::displaced_typed_blocks<
      MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator>& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable<
        ::yampi::displaced_typed_blocks<
         MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> >::value ))
  { lhs.swap(rhs); }


  namespace derived_datatype_detail
  {
    /*
    template <typename Value, typename Member, typename... Members>
    inline ::yampi::datatype do_derive(
      ::yampi::datatype const& datatype, Value const& value,
      ::yampi::environment const& environment,
      Member const& member, Members const&... members)
    {
      MPI_Aint value_address = ::yampi::address(value, environment);
      YAMPI_array<int, sizeof...(Members)+2u> block_lengths;
      YAMPI_array<MPI_Aint, sizeof...(Members)+2u> block_lengths;
      YAMPI_array<MPI_Datatype, sizeof...(Members)+2u> block_lengths;
    }


    template <typename Value>
    struct derive
    {
      static ::yampi::datatype call(
        ::yampi::datatype const& datatype, Value const& value,
        ::yampi::environment const& environment)
      { return ::yampi::access::derive(datatype, value, environment); }
    };

    template <typename Real>
    struct derive< std::complex<Real> >
    {
      static ::yampi::datatype call(
        ::yampi::datatype const& datatype, std::complex<Real> const& value,
        ::yampi::environment const& environment)
      {
        return ::yampi::derived_datatype_detail::do_derive(
          datatype, value, environment, value.real(), value.imag());
      }
    };
    */
  }


  class derived_datatype
  {
    ::yampi::datatype datatype_;
    bool is_free_;

   public:
    typedef MPI_Datatype mpi_datatype_type;
    typedef MPI_Aint mpi_address_type;

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    derived_datatype() = default;
# else
    derived_datatype() : datatype_(), is_free_(true) { }
# endif

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    derived_datatype(derived_datatype const&) = delete;
    derived_datatype& operator=(derived_datatype const&) = delete;
# else
   private:
    derived_datatype(derived_datatype const&);
    derived_datatype& operator=(derived_datatype const&);

   public:
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    derived_datatype(derived_datatype&&) = default;
    derived_datatype& operator=(derived_datatype&&) = default;
#   else
    derived_datatype(derived_datatype&& other) BOOST_NOEXCEPT
      : datatype_(std::move(other.datatype_)), is_free_(std::move(other.is_free_))
    { }

    derived_datatype& operator=(derived_datatype&& other) BOOST_NOEXCEPT
    {
      if (this != &other)
      {
        datatype_ = std::move(other.datatype_);
        is_free_ = std::move(other.is_free_);
      }
      return *this;
    }
#   endif
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~derived_datatype() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (is_free_)
        return;

      //MPI_Type_free(YAMPI_addressof(datatype_.mpi_datatype()));
      MPI_Datatype mpi_datatype = datatype_.mpi_datatype();
      MPI_Type_free(YAMPI_addressof(mpi_datatype));
    }

    // MPI_Type_contiguous
    derived_datatype(
      ::yampi::datatype const& base_datatype, int const count,
      ::yampi::environment const& environment)
      : datatype_(derive(base_datatype, count, environment)),
        is_free_(false)
    { }

    // MPI_Type_vector
    derived_datatype(
      ::yampi::datatype const& base_datatype, ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment)
      : datatype_(derive(base_datatype, block, count, environment)),
        is_free_(false)
    { }

    // MPI_Type_indexed
    template <typename Allocator>
    derived_datatype(
      ::yampi::datatype const& base_datatype,
      ::yampi::displaced_blocks<Allocator> const& blocks,
      ::yampi::environment const& environment)
      : datatype_(derive(base_datatype, blocks, environment)),
        is_free_(false)
    { }

    // MPI_Type_create_indexed_block
    template <typename Allocator>
    derived_datatype(
      ::yampi::datatype const& base_datatype,
      ::yampi::displaced_constant_blocks<Allocator> const& blocks,
      ::yampi::environment const& environment)
      : datatype_(derive(base_datatype, blocks, environment)),
        is_free_(false)
    { }

    // MPI_Type_create_struct
    template <
      typename MpiDatatypeAllocator, typename BlockLengthAllocator, typename MpiAddressAllocator>
    derived_datatype(
      ::yampi::displaced_typed_blocks<
        MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& blocks,
      ::yampi::environment const& environment)
      : datatype_(derive(blocks, environment)),
        is_free_(false)
    { }


    void free(::yampi::environment const& environment)
    {
      //int const error_code = MPI_Type_free(YAMPI_addressof(datatype_.mpi_datatype()));
      MPI_Datatype mpi_datatype = datatype_.mpi_datatype();
      int const error_code = MPI_Type_free(YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "::yampi::derived_datatype::free", environment);

      is_free_ = true;
      datatype_ = yampi::datatype(mpi_datatype);
    }

    ::yampi::datatype const& datatype() const { return datatype_; }


   private:
    ::yampi::datatype commit(MPI_Datatype& mpi_datatype, ::yampi::environment const& environment)
    {
      int const error_code = MPI_Type_commit(YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "::yampi::derived_datatype::commit", environment);

      return ::yampi::datatype(mpi_datatype);
    }


    // MPI_Type_contiguous
    ::yampi::datatype derive(
      ::yampi::datatype const& base_datatype, int const count,
      ::yampi::environment const& environment)
    {
      MPI_Datatype mpi_datatype;
      int const error_code
        = MPI_Type_contiguous(count, base_datatype.mpi_datatype(), YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "::yampi::derived_datatype::derive_contiguously", environment);

      return commit(mpi_datatype, environment);
    }

    // MPI_Type_vector
    ::yampi::datatype derive(
      ::yampi::datatype const& base_datatype,
      ::yampi::strided_block const& block, int const count,
      ::yampi::environment const& environment)
    {
      MPI_Datatype mpi_datatype;
      int const error_code
        = MPI_Type_vector(
            count, block.length(), block.stride(),
            base_datatype.mpi_datatype(), YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "::yampi::derived_datatype::derive_strided_blocks", environment);

      return commit(mpi_datatype, environment);
    }

    // MPI_Type_indexed
    template <typename Allocator>
    ::yampi::datatype derive(
      ::yampi::datatype const& base_datatype,
      ::yampi::displaced_blocks<Allocator> const& blocks,
      ::yampi::environment const& environment)
    {
      MPI_Datatype mpi_datatype;
      int const error_code
        = MPI_Type_indexed(
            static_cast<int>(blocks.size()),
            const_cast<int*>(YAMPI_addressof(blocks.block_lengths().front())),
            const_cast<int*>(YAMPI_addressof(blocks.displacements().front())),
            base_datatype.mpi_datatype(), YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "::yampi::derived_datatype::derive_displaced_blocks", environment);

      return commit(mpi_datatype, environment);
    }

    // MPI_Type_create_indexed_block
    template <typename Allocator>
    ::yampi::datatype derive(
      ::yampi::datatype const& base_datatype,
      ::yampi::displaced_constant_blocks<Allocator> const& blocks,
      ::yampi::environment const& environment)
    {
      MPI_Datatype mpi_datatype;
      int const error_code
        = MPI_Type_create_indexed_block(
            static_cast<int>(blocks.size()), blocks.block_length(),
            const_cast<int*>(YAMPI_addressof(blocks.displacements().front())),
            base_datatype.mpi_datatype(), YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "::yampi::derived_datatype::derive_displaced_blocks", environment);

      return commit(mpi_datatype, environment);
    }

    // MPI_Type_create_struct
    template <
      typename MpiDatatypeAllocator, typename BlockLengthAllocator, typename MpiAddressAllocator>
    ::yampi::datatype derive(
      ::yampi::displaced_typed_blocks<
        MpiDatatypeAllocator, BlockLengthAllocator, MpiAddressAllocator> const& blocks,
      ::yampi::environment const& environment)
    {
      MPI_Datatype mpi_datatype;
      int const error_code
        = MPI_Type_create_struct(
            static_cast<int>(blocks.size()),
            const_cast<int*>(YAMPI_addressof(blocks.block_lengths().front())),
            const_cast<MPI_Aint*>(YAMPI_addressof(blocks.displacement_addresses().front())),
            const_cast<MPI_Datatype*>(YAMPI_addressof(blocks.mpi_datatypes().front())),
            YAMPI_addressof(mpi_datatype));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(
          error_code, "::yampi::derived_datatype::derive_displaced_blocks", environment);

      return commit(mpi_datatype, environment);
    }


    // TODO: implement easier versions of MPI_Type_create_struct
    /*
    template <typename Value>
    ::yampi::datatype derive(
      ::yampi::datatype const& datatype, Value const& value,
      ::yampi::environment const& environment)
    {
      ::yampi::datatype new_datatype
        = ::yampi::derived_datatype_detail::derive<Value>::call(
            datatype, value, environment);
      commit(new_datatype.mpi_datatype(), environment);
      datatype_.push_back(new_datatype.mpi_datatype());
      return new_datatype;
    }

    template <typename Value>
    ::yampi::datatype derive(
      ::yampi::datatype const& datatype, ::yampi::environment const& environment)
    {
      ::yampi::datatype new_datatype
        = ::yampi::derived_datatype_detail::derive<Value>::call(
            datatype, Value(), environment);
      commit(new_datatype.mpi_datatype(), environment);
      datatype_.push_back(new_datatype.mpi_datatype());
      return new_datatype;
    }

    template <typename Value, typename Member, typename... Members>
    ::yampi::datatype derive(
      ::yampi::datatype const& datatype, Value const& value,
      ::yampi::environment const& environment,
      Member const& member, Members const&... members)
    {
      ::yampi::datatype new_type
        = ::yampi::derived_datatype_detail::do_derive(
            datatype, value, environment, member, members...);
      commit(new_datatype.mpi_datatype(), environment);
      datatype_.push_back(new_datatype.mpi_datatype());
      return new_datatype;
    }
    */
  };
}


# undef YAMPI_addressof

#endif
