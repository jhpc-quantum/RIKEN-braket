#ifndef YAMPI_ALGORITHM_COPY_HPP
# define YAMPI_ALGORITHM_COPY_HPP

# include <boost/config.hpp>

# include <cassert>

# ifdef __FUJITSU // needed for combination of Boost 1.61.0 and Fujitsu compiler
#   include <boost/utility/in_place_factory.hpp>
#   include <boost/utility/typed_in_place_factory.hpp>
# endif
# include <boost/optional.hpp>

# include <yampi/blocking_send.hpp>
# include <yampi/blocking_receive.hpp>
# include <yampi/environment.hpp>
# include <yampi/algorithm/ranked_buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>


namespace yampi
{
  namespace algorithm
  {
    template <typename Value>
    inline boost::optional< ::yampi::status>
    copy(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank();

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::blocking_receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::blocking_send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }

    template <typename Value>
    inline boost::optional< ::yampi::status>
    copy(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank();

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::blocking_receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
        ::yampi::blocking_send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);

      return boost::none;
    }


    // ignoring status
    template <typename Value>
    inline void copy(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank();

      if (present_rank == receive_buffer.rank())
        ::yampi::blocking_receive(
          communicator, ::yampi::ignore_status(), environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::blocking_send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }

    template <typename Value>
    inline void copy(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank();

      if (present_rank == receive_buffer.rank())
        ::yampi::blocking_receive(
          communicator, ::yampi::ignore_status(), environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
        ::yampi::blocking_send(
          communicator, environment, send_buffer.buffer(), receive_buffer.rank(), tag);
    }
  }
}


#endif

