#ifndef YAMPI_LOWEST_IO_PROCESS_HPP
# define YAMPI_LOWEST_IO_PROCESS_HPP

# include <boost/config.hpp>

# ifdef __FUJITSU // needed for combination of Boost 1.61.0 and Fujitsu compiler
#   include <boost/utility/in_place_factory.hpp>
#   include <boost/utility/typed_in_place_factory.hpp>
# endif
# include <boost/optional.hpp>
# include <boost/none.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/binary_operation.hpp>


namespace yampi
{
  // return the rank that is lowest in the group of world communicator
  inline boost::optional< ::yampi::rank > lowest_io_process(
    ::yampi::environment const& environment)
  {
    BOOST_CONSTEXPR_OR_CONST ::yampi::rank zero_rank = ::yampi::rank(0);
    ::yampi::rank const io_process = ::yampi::io_process(environment);
    if (io_process == ::yampi::any_source())
      return zero_rank;
    else if (io_process == ::yampi::null_process())
      return boost::none;

    return static_cast< ::yampi::rank >(
      ::yampi::all_reduce(
        ::yampi::world_communicator(), environment,
        ::yampi::make_buffer(io_process.mpi_rank()),
        ::yampi::operations::minimum()));
  }
}


#endif
