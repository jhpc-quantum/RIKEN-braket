#ifndef YAMPI_IS_INITIALIZED_HPP
# define YAMPI_IS_INITIALIZED_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  class is_initialized_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit is_initialized_error(int const error_code)
     : std::runtime_error("Error occurred when checking initialized"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };


  inline bool is_initialized()
  {
    int result;
    int const error_code = MPI_Initialized(&result);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::is_initialized_error(error_code);

    return result;
  }
}


#endif

