#ifndef YAMPI_COMPARE_HPP
# define YAMPI_COMPARE_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class communicators_are
    : int
  {
    identical = MPI_IDENT, congruent = MPI_CONGRUENT,
    similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
  };
# else
  enum communicators_are
  {
    identical = MPI_IDENT, congruent = MPI_CONGRUENT,
    similar = MPI_SIMILAR, unequal = MPI_UNEQUAL
  };
# endif

  inline ::yampi::communicators_are compare(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  {
    int result;
    int const error_code = MPI_Comm_compare(lhs.mpi_comm(), rhs.mpi_comm(), &result);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::compare", environment);
    return static_cast<communicators_are>(result);
  }

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  inline bool is_identical(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::identical; }

  inline bool is_congruent(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::congruent; }

  inline bool is_similar(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::similar; }

  inline bool is_unequal(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::communicators_are::unequal; }
# else
  inline bool is_identical(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::identical; }

  inline bool is_congruent(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::congruent; }

  inline bool is_similar(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::similar; }

  inline bool is_unequal(
    ::yampi::communicator const lhs, ::yampi::communicator const rhs,
    ::yampi::environment const& environment)
  { return ::yampi::compare(lhs, rhs, environment) == ::yampi::unequal; }
# endif
}


#endif

