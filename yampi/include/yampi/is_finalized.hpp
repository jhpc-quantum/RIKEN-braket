#ifndef YAMPI_IS_FINALIZED_HPP
# define YAMPI_IS_FINALIZED_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  class is_finalized_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit is_finalized_error(int const error_code)
     : std::runtime_error("Error occurred when checking finalized"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };


  inline bool is_finalized()
  {
    int result;
    int const error_code = MPI_Finalized(&result);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::is_finalized_error(error_code);

    return result;
  }
}


#endif

