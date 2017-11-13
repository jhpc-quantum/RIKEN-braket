#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

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
  class broadcast
  {
    ::yampi::communicator communicator_;
    ::yampi::rank root_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    broadcast() = delete;
# else
   private:
    broadcast();

   public:
# endif

    BOOST_CONSTEXPR broadcast(
      ::yampi::communicator const communicator, ::yampi::rank const root)
      BOOST_NOEXCEPT_OR_NOTHROW
      : communicator_(communicator), root_(root)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    broadcast(broadcast const&) = default;
    broadcast& operator=(broadcast const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    broadcast(broadcast&&) = default;
    broadcast& operator=(broadcast&&) = default;
#   endif
    ~broadcast() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename Value>
    void call(::yampi::environment const& environment, ::yampi::buffer<Value>& buffer) const
    {
      int const error_code
        = MPI_Bcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }

    template <typename Value>
    void call(::yampi::environment const& environment, ::yampi::buffer<Value> const& buffer) const
    {
      int const error_code
        = MPI_Bcast(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }
  };
}


#endif

