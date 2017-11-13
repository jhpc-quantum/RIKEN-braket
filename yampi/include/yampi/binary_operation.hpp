#ifndef YAMPI_BINARY_OPERATION_HPP
# define YAMPI_BINARY_OPERATION_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  struct maximum_t { };
  struct minimum_t { };
  struct plus_t { };
  struct multiplies_t { };
  struct logical_and_t { };
  struct bit_and_t { };
  struct logical_or_t { };
  struct bit_or_t { };
  struct logical_xor_t { };
  struct bit_xor_t { };
  struct maximum_location_t { };
  struct minimum_location_t { };

  class binary_operation
  {
    MPI_Op mpi_op_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    binary_operation() = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    binary_operation();

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

    explicit BOOST_CONSTEXPR binary_operation(MPI_Op const mpi_op) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_op_(mpi_op)
    { }

# ifndef __FUJITSU
#   define YAMPI_CONSTEXPR BOOST_CONSTEXPR
# else
#   define YAMPI_CONSTEXPR
# endif
# define YAMPI_DEFINE_OPERATION_CONSTRUCTOR(op, mpiop) \
    explicit YAMPI_CONSTEXPR binary_operation(::yampi:: op ## _t const) BOOST_NOEXCEPT_OR_NOTHROW\
      : mpi_op_(MPI_ ## mpiop )\
    { }

    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(maximum, MAX)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(minimum, MIN)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(plus, SUM)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(multiplies, PROD)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_and, LAND)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_and, BAND)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_or, LOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_or, BOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(logical_xor, LXOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(bit_xor, BXOR)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(maximum_location, MAXLOC)
    YAMPI_DEFINE_OPERATION_CONSTRUCTOR(minimum_location, MINLOC)

# undef YAMPI_DEFINE_OPERATION_CONSTRUCTOR
# undef YAMPI_CONSTEXPR

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    binary_operation(binary_operation const&) = default;
    binary_operation& operator=(binary_operation const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    binary_operation(binary_operation&&) = default;
    binary_operation& operator=(binary_operation&&) = default;
#   endif
    ~binary_operation() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR bool operator==(binary_operation const other) const
    { return mpi_op_ == other.mpi_op_; }

    BOOST_CONSTEXPR MPI_Op const& mpi_op() const { return mpi_op_; }

    void swap(binary_operation& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Op>::value ))
    {
      using std::swap;
      swap(mpi_op_, other.mpi_op_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(
    ::yampi::binary_operation const lhs, ::yampi::binary_operation const rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::binary_operation& lhs, ::yampi::binary_operation& rhs)
    BOOST_NOEXCEPT_IF((
      ::yampi::utility::is_nothrow_swappable< ::yampi::binary_operation >::value ))
  { lhs.swap(rhs); }


  namespace operations
  {
# ifndef __FUJITSU
#   define YAMPI_CONSTEXPR BOOST_CONSTEXPR
# else
#   define YAMPI_CONSTEXPR
# endif
# define YAMPI_DEFINE_OPERATION_FUNCTION(op) \
    inline YAMPI_CONSTEXPR ::yampi::binary_operation op ()\
    { return ::yampi::binary_operation(::yampi:: op ## _t()); }

    YAMPI_DEFINE_OPERATION_FUNCTION(maximum)
    YAMPI_DEFINE_OPERATION_FUNCTION(minimum)
    YAMPI_DEFINE_OPERATION_FUNCTION(plus)
    YAMPI_DEFINE_OPERATION_FUNCTION(multiplies)
    YAMPI_DEFINE_OPERATION_FUNCTION(logical_and)
    YAMPI_DEFINE_OPERATION_FUNCTION(bit_and)
    YAMPI_DEFINE_OPERATION_FUNCTION(logical_or)
    YAMPI_DEFINE_OPERATION_FUNCTION(bit_or)
    YAMPI_DEFINE_OPERATION_FUNCTION(logical_xor)
    YAMPI_DEFINE_OPERATION_FUNCTION(bit_xor)
    YAMPI_DEFINE_OPERATION_FUNCTION(maximum_location)
    YAMPI_DEFINE_OPERATION_FUNCTION(minimum_location)

# undef YAMPI_DEFINE_OPERATION_FUNCTION
# undef YAMPI_CONSTEXPR
  }
}


#endif
