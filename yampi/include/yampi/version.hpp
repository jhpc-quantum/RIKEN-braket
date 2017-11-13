#ifndef YAMPI_VERSION_HPP
# define YAMPI_VERSION_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class version_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit version_error(int const error_code)
     : std::runtime_error("Error occurred when getting version"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };


  class version_t
  {
    int major_;
    int minor_;

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    BOOST_CONSTEXPR version_t() BOOST_NOEXCEPT_OR_NOTHROW = default;
    version_t(version_t const&) = default;
    version_t& operator=(version_t const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    version_t(version_t&&) = default;
    version_t& operator=(version_t&&) = default;
#   endif
    ~version_t() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR version_t(int const major, int const minor) BOOST_NOEXCEPT_OR_NOTHROW
      : major_(major), minor_(minor)
    { }

    int major() const { return major_; }
    int minor() const { return minor_; }

    void swap(version_t& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<int>::value ))
    {
      using std::swap;
      swap(major_, other.major_);
      swap(minor_, other.minor_);
    }
  };

  inline bool operator==(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  { return lhs.major() == rhs.major() and lhs.minor() == rhs.minor(); }

  inline bool operator!=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  { return not (lhs == rhs); }

  inline bool operator<(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  {
    return lhs.major() < rhs.major()
      or (lhs.major() == rhs.major() and lhs.minor() < rhs.minor());
  }

  inline bool operator>=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  { return rhs < lhs; }

  inline bool operator<=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs)
  { return not (rhs < lhs); }

  inline void swap(::yampi::version_t& lhs, ::yampi::version_t& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::version_t >::value ))
  { lhs.swap(rhs); }


  inline ::yampi::version_t version()
  {
    int major, minor;
    int const error_code = MPI_Get_version(&major, &minor);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::version_error(error_code);

    return ::yampi::version_t(major, minor);
  }
}


#endif

