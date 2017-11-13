#ifndef YAMPI_DATATYPE_HPP
# define YAMPI_DATATYPE_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class datatype
  {
    MPI_Datatype mpi_datatype_;

   public:
# ifndef __FUJITSU
#   define YAMPI_CONSTEXPR BOOST_CONSTEXPR
# else
#   define YAMPI_CONSTEXPR
# endif
    YAMPI_CONSTEXPR datatype() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(MPI_DATATYPE_NULL)
    { }

    explicit BOOST_CONSTEXPR datatype(MPI_Datatype const& mpi_datatype) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_datatype_(mpi_datatype)
    { }
# undef YAMPI_CONSTEXPR

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    datatype(datatype const&) = default;
    datatype& operator=(datatype const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    datatype(datatype&&) = default;
    datatype& operator=(datatype&&) = default;
#   endif
    ~datatype() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    bool operator==(datatype const& other) const { return mpi_datatype_ == other.mpi_datatype_; }

    MPI_Datatype const& mpi_datatype() const { return mpi_datatype_; }

    void swap(datatype& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Datatype>::value ))
    {
      using std::swap;
      swap(mpi_datatype_, other.mpi_datatype_);
    }
  };

  inline bool operator!=(::yampi::datatype const& lhs, ::yampi::datatype const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::datatype& lhs, ::yampi::datatype& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::datatype >::value ))
  { lhs.swap(rhs); }
}


#endif

