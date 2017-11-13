#ifndef YAMPI_ALLOCATOR_HPP
# define YAMPI_ALLOCATOR_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# include <limits>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif
# if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#   include <utility>
# endif

# include <mpi.h>

# include <yampi/is_initialized.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_true_type std::true_type
# else
#   define YAMPI_true_type boost::true_type
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class allocate_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit allocate_error(int const error_code)
     : std::runtime_error("Error occurred when allocating"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };

  class deallocate_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit deallocate_error(int const error_code)
     : std::runtime_error("Error occurred when deallocating"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };


  // TODO: modify to depend on yampi::environment
  template <typename T>
  class allocator
  {
   public:
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    typedef YAMPI_true_type propagate_on_container_move_assignment;
    typedef YAMPI_true_type is_always_equal;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    BOOST_CONSTEXPR allocator() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else
    BOOST_CONSTEXPR allocator() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif

    allocator(allocator const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { }

    template <typename U>
    allocator(allocator<U> const& other) BOOST_NOEXCEPT_OR_NOTHROW
    { }

    pointer address(reference x) const { return YAMPI_addressof(x); }
    const_pointer address(const_reference x) const { return YAMPI_addressof(x); }

    pointer allocate(std::size_t const n, const_void_pointer = nullptr)
    {
      assert(::yampi::is_initialized());

      pointer result;
      int const error_code
        = MPI_Alloc_mem(static_cast<MPI_Aint>(n * sizeof(T)), MPI_INFO_NULL, &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::allocate_error(error_code);

      return result;
    }

    void deallocate(pointer ptr, std::size_t const)
    {
      int const error_code = MPI_Free_mem(ptr);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::deallocate_error(error_code);
    }

    size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW
    { return std::numeric_limits<std::size_t>::max() / sizeof(T); }

# if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
    template <typename U, typename... Arguments>
    void construct(U* ptr, Arguments&&... arguments)
    { ::new((void *)ptr) U(std::forward<Arguments>(arguments)...); }
# else
    void construct(pointer ptr, const_reference value)
    { new((void *)ptr) T(value); }
# endif

    void destroy(pointer ptr) { ((T*)ptr)->~T(); }
    template <typename U>
    void destroy(U* ptr) { ptr->~U(); }
  };

  template <>
  class allocator<void>
  {
   public:
    typedef void* pointer;
    typedef void const* const_pointer;
    typedef void* void_pointer;
    typedef void const* const_void_pointer;
    typedef void value_type;

    template <typename U>
    struct rebind
    { typedef allocator<U> other; };
  };


  template <typename T, typename U>
  inline BOOST_CONSTEXPR bool operator==(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) BOOST_NOEXCEPT_OR_NOTHROW
  { return true; }

  template <typename T, typename U>
  inline BOOST_CONSTEXPR bool operator!=(::yampi::allocator<T> const&, ::yampi::allocator<U> const&) BOOST_NOEXCEPT_OR_NOTHROW
  { return false; }
}


# undef YAMPI_true_type
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef YAMPI_addressof

#endif

