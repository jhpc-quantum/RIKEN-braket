#ifndef YAMPI_BLOCKING_SEND_HPP
# define YAMPI_BLOCKING_SEND_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename Value>
  inline void blocking_send(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    int const error_code
      = MPI_Send(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_send", environment);
  }
}


#endif

