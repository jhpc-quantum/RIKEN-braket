#ifndef YAMPI_BLOCKING_RECEIVE_HPP
# define YAMPI_BLOCKING_RECEIVE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  // Blocking receive
  template <typename Value>
  inline ::yampi::status blocking_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename Value>
  inline ::yampi::status blocking_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Recv(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive", environment);

    return ::yampi::status(stat);
  }

  // Blocking receive (ignoring status)
  template <typename Value>
  inline void blocking_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive", environment);
  }

  template <typename Value>
  inline void blocking_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Recv(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_receive", environment);
  }
}


# undef YAMPI_addressof

#endif

