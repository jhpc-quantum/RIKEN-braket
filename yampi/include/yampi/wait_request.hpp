#ifndef YAMPI_WAIT_REQUEST_HPP
# define YAMPI_WAIT_REQUEST_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/status.hpp>
# include <yampi/eror.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  inline ::yampi::status wait_request(
    ::yampi::request const& request, ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Wait(YAMPI_addressof(request.mpi_request()), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait", environment);

    return ::yampi::status(stat);
  }

  inline void wait_request(
    ::yampi::request const& request,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment)
  {
    MPI_Status stat;
    int const error_code
      = MPI_Wait(YAMPI_addressof(request.mpi_request()), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::wait", environment);
  }
}


# undef YAMPI_addressof

#endif

