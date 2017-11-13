#ifndef YAMPI_GATHER_TO_ALL_HPP
# define YAMPI_GATHER_TO_ALL_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
# endif
# include <iterator>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  // TODO: implement MPI_Allgatherv
  template <typename SendValue, typename ContiguousIterator>
  inline void gather_to_all(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ContiguousIterator const first)
  {
    static_assert(
      (YAMPI_is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");

    int const error_code
      = MPI_Allgather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<SendValue*>(YAMPI_addressof(*first)),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather_to_all", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void gather_to_all(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer)
  {
    int const error_code
      = MPI_Allgather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          YAMPI_addressof(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather_to_all", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void gather_to_all(
    ::yampi::communicator const communicator, ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer)
  {
    int const error_code
      = MPI_Allgather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(YAMPI_addressof(receive_buffer.data())),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather_to_all", environment);
  }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif
