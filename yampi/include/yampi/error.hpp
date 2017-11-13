#ifndef YAMPI_ERROR_HPP
# define YAMPI_ERROR_HPP

# include <boost/config.hpp>

# include <string>
# include <stdexcept>

# include <mpi.h>


namespace yampi
{
  class error_in_error
    : public std::runtime_error
  {
   public:
    explicit error_in_error(std::string const& where)
      : std::runtime_error(
          (std::string("In ") + where + ": error occured in MPI_Error_class/string").c_str())
    { }
  };

  class environment;

  class error
    : public std::runtime_error
  {
    int error_class_;

   public:
    error(
      int const error_code, std::string const& where, ::yampi::environment const& environment)
     : std::runtime_error(generate_what_string(error_code, where, environment).c_str()),
       error_class_(generate_error_class(error_code, where, environment))
    { }

    int error_class() const { return error_class_; }

   private:
    int generate_error_class(
      int const error_code, std::string const& where, ::yampi::environment const&) const
    {
      int error_class;
      int const error_class_error_code = MPI_Error_class(error_code, &error_class);

      if (error_class_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error(where);

      return error_class;
    }

    std::string generate_what_string(
      int const error_code, std::string const& where, ::yampi::environment const&) const
    {
      char error[MPI_MAX_ERROR_STRING];
      int length;
      int const error_string_error_code = MPI_Error_string(error_code, error, &length);

      if (error_string_error_code != MPI_SUCCESS)
        throw ::yampi::error_in_error(where);

      return std::string("In ") + where + ": " + std::string(error);
    }
  };
}


#endif

