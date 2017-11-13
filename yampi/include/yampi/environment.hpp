#ifndef YAMPI_ENVIRONMENT_HPP
# define YAMPI_ENVIRONMENT_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <stdexcept>

# include <mpi.h>

# include <yampi/is_initialized.hpp>
# include <yampi/is_finalized.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class initialization_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit initialization_error(int const error_code)
      : std::runtime_error("Error occurred when initializing"),
        error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };

  class already_initialized_error
    : public std::logic_error
  {
   public:
    already_initialized_error()
      : std::logic_error("MPI environment has been already initialized")
    { }
  };

  class already_finalized_error
    : public std::logic_error
  {
   public:
    already_finalized_error()
      : std::logic_error("MPI environment has been already finalized")
    { }
  };


  class environment
  {
   public:
    BOOST_STATIC_CONSTEXPR int major_version = MPI_VERSION;
    BOOST_STATIC_CONSTEXPR int minor_version = MPI_SUBVERSION;

    environment()
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int const error_code = MPI_Init(NULL, NULL);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);
    }

    environment(int argc, char* argv[])
    {
      if (::yampi::is_initialized())
        throw ::yampi::already_initialized_error();

      int const error_code = MPI_Init(&argc, &argv);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::initialization_error(error_code);
    }

    ~environment() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (::yampi::is_finalized())
        return;

      MPI_Finalize();
    }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    environment(environment const&) = delete;
    environment& operator=(environment const&) = delete;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    environment(environment&&) = delete;
    environment& operator=(environment&&) = delete;
#   endif
# else
   private:
    environment(environment const&);
    environment& operator=(environment const&);
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    environment(environment&&);
    environment& operator=(environment&&);
#   endif

   public:
# endif

    void finalize()
    {
      if (::yampi::is_finalized())
        throw ::yampi::already_finalized_error();

      int const error_code = MPI_Finalize();
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::environment::finalize", *this);
    }
  };
}


#endif

