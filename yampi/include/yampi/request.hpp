#ifndef YAMPI_REQUEST_HPP
# define YAMPI_REQUEST_HPP

# include <boost/config.hpp>

# include <utility>

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>


namespace yampi
{
  class null_request_t { };

  class request
  {
    MPI_Request mpi_request_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request() = delete;
# else
   private:
    request();

   public:
# endif

    explicit request(MPI_Request const& req) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_request_(req)
    { }

    explicit request(::yampi::null_request_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_request_(MPI_REQUEST_NULL)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    request(request const&) = default;
    request& operator=(request const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    request(request&&) = default;
    request& operator=(request&&) = default;
#   endif
    ~request() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    bool operator==(request const& other) const
    { return mpi_request_ == other.mpi_request_; }

    MPI_Request const& mpi_request() const { return mpi_request_; }

    void swap(request& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Request>::value ))
    {
      using std::swap;
      swap(mpi_request_, other.mpi_request_);
    }
  };

  inline bool operator!=(::yampi::request const& lhs, ::yampi::request const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::request& lhs, ::yampi::request& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::requst >::value ))
  { lhs.swap(rhs); }


  inline ::yampi::request null_request()
  {
    static ::yampi::request const request(::yampi::null_request_t());
    return request;
  }

  inline bool is_valid_request(::yampi::request const& self)
  { return self != ::yampi::null_request(); }
}


#endif
