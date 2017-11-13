#ifndef YAMPI_ALGORITHM_SWAP_HPP
# define YAMPI_ALGORITHM_SWAP_HPP

# include <boost/config.hpp>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/send_receive.hpp>
# include <yampi/status.hpp>
# include <yampi/tag.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue>& receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive(
        communicator, environment, send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag);
    }

    template <typename SendValue, typename ReceiveValue>
    inline ::yampi::status swap(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue> const& receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive(
        communicator, environment, send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag);
    }


    // with replacement
    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive(
        communicator, environment, buffer, swap_rank, tag, swap_rank, tag);
    }

    template <typename Value>
    inline ::yampi::status swap(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      return ::yampi::send_receive(
        communicator, environment, buffer, swap_rank, tag, swap_rank, tag);
    }


    // ignoring status
    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue>& receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive(
        communicator, ::yampi::ignore_status(), environment,
        send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag);
    }

    template <typename SendValue, typename ReceiveValue>
    inline void swap(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue> const& receive_buffer,
      ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive(
        communicator, ::yampi::ignore_status(), environment,
        send_buffer, swap_rank, tag, receive_buffer, swap_rank, tag);
    }


    // with replacement, ignoring status
    template <typename Value>
    inline void swap(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::buffer<Value>& buffer, ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive(
        communicator, ::yampi::ignore_status(), environment,
        buffer, swap_rank, tag, swap_rank, tag);
    }

    template <typename Value>
    inline void swap(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer, ::yampi::rank const swap_rank,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      ::yampi::send_receive(
        communicator, ::yampi::ignore_status(), environment,
        buffer, swap_rank, tag, swap_rank, tag);
    }
  }
}


#endif

