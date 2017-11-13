#ifndef YAMPI_PROCESSOR_NAME_HPP
# define // YAMPI_PROCESSOR_NAME_HPP

# include <boost/config.hpp>

# include <string>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  inline std::string processor_name(::yampi::environment const& environment)
  {
    char name[MPI_MAX_PROCESSOR_NAME];
    int length;
    int const error_code = MPI_Get_processor_name(name, &length);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::environment::processor_name", environment);

    return name;
  }
}


#endif
