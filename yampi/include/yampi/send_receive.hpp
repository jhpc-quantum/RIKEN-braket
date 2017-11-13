#ifndef YAMPI_SEND_RECEIVE_HPP
# define YAMPI_SEND_RECEIVE_HPP

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
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }


  // with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    MPI_Status stat;
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);

    return ::yampi::status(stat);
  }


  // ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }


  // with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::communicator const communicator,
    ::yampi::ignore_status_t const, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const receive_tag = ::yampi::any_tag())
  {
    int const error_code
      = MPI_Sendrecv_replace(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send_receive", environment);
  }
}


# undef YAMPI_addressof

#endif

