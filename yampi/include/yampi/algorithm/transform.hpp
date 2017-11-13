#ifndef YAMPI_ALGORITHM_TRANSFORM_HPP
# define YAMPI_ALGORITHM_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# include <vector>
# include <algorithm>

# ifdef __FUJITSU // needed for combination of Boost 1.61.0 and Fujitsu compiler
#   include <boost/utility/in_place_factory.hpp>
#   include <boost/utility/typed_in_place_factory.hpp>
# endif
# include <boost/optional.hpp>
# include <boost/none.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# include <yampi/blocking_send.hpp>
# include <yampi/blocking_receive.hpp>
# include <yampi/allocator.hpp>
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
    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status>
    transform(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::blocking_receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::blocking_send(
          communicator, environment,
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag);
      }

      return boost::none;
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status>
    transform(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        communicator, environment,
        send_buffer, receive_buffer, transform_buffer, unary_function, tag);
    }

    template <typename Value, typename Allocator, typename UnaryFunction>
    inline boost::optional< ::yampi::status>
    transform(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return boost::none;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        return boost::make_optional(
          ::yampi::blocking_receive(
            communicator, environment, receive_buffer.buffer(), send_buffer.rank(), tag));
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::blocking_send(
          communicator, environment,
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag);
      }

      return boost::none;
    }

    template <typename Value, typename UnaryFunction>
    inline boost::optional< ::yampi::status>
    transform(
      ::yampi::communicator const communicator, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        communicator, environment,
        send_buffer, receive_buffer, transform_buffer, unary_function, tag);
    }


    // ignoring status
    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::blocking_receive(
          communicator, ::yampi::ignore_status(), environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::blocking_send(
          communicator, environment,
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag);
      }
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value>& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        communicator, ::yampi::ignore_status(), environment,
        send_buffer, receive_buffer, transform_buffer, unary_function, tag);
    }
    template <typename Value, typename Allocator, typename UnaryFunction>
    inline void transform(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      std::vector<Value, Allocator>& transform_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      assert(send_buffer.count() == receive_buffer.count());

      if (send_buffer.rank() == receive_buffer.rank())
        return;

      ::yampi::rank const present_rank = communicator.rank(environment);

      if (present_rank == receive_buffer.rank())
        ::yampi::blocking_receive(
          communicator, ::yampi::ignore_status(), environment,
          receive_buffer.buffer(), send_buffer.rank(), tag);
      else if (present_rank == send_buffer.rank())
      {
        transform_buffer.clear();
        transform_buffer.reserve(send_buffer.count());
        std::transform(
          send_buffer.data(), send_buffer.data()+send_buffer.count(),
          std::back_inserter(transform_buffer), unary_function);

        ::yampi::blocking_send(
          communicator, environment,
          ::yampi::make_buffer(
            boost::begin(transform_buffer), boost::end(transform_buffer), send_buffer.datatype()),
          receive_buffer.rank(), tag);
      }
    }

    template <typename Value, typename UnaryFunction>
    inline void transform(
      ::yampi::communicator const communicator,
      ::yampi::ignore_status_t const, ::yampi::environment const& environment,
      ::yampi::algorithm::ranked_buffer<Value> const& send_buffer,
      ::yampi::algorithm::ranked_buffer<Value> const& receive_buffer,
      UnaryFunction unary_function,
      ::yampi::tag const tag = ::yampi::tag(0))
    {
      std::vector<Value, ::yampi::allocator<Value> > transform_buffer;
      return ::yampi::algorithm::transform(
        communicator, ::yampi::ignore_status(), environment,
        send_buffer, receive_buffer, transform_buffer, unary_function, tag);
    }
  }
}


# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_enable_if
# undef YAMPI_is_same

#endif

