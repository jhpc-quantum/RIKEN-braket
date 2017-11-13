#ifndef YAMPI_ALGORITHM_RANKED_BUFFER_HPP
# define YAMPI_ALGORITHM_RANKED_BUFFER_HPP

# include <boost/config.hpp>

# include <iterator>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/rank.hpp>
# include <yampi/datatype.hpp>
# include <yampi/buffer.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_cv std::remove_cv
# else
#   define YAMPI_remove_cv boost::remove_cv
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  namespace algorithm
  {
    template <typename T>
    class ranked_buffer
    {
      ::yampi::buffer<T> buffer_;
      ::yampi::rank rank_;

     public:
      ranked_buffer(T& value, ::yampi::rank const rank)
        : buffer_(value), rank_(rank)
      { }

      ranked_buffer(T const& value, ::yampi::rank const rank)
        : buffer_(value), rank_(rank)
      { }

      ranked_buffer(T& value, ::yampi::datatype const datatype, ::yampi::rank const rank)
        : buffer_(value, datatype), rank_(rank)
      { }

      ranked_buffer(T const& value, ::yampi::datatype const datatype, ::yampi::rank const rank)
        : buffer_(value, datatype), rank_(rank)
      { }

      template <typename ContiguousIterator>
      ranked_buffer(
        ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const rank)
        : buffer_(first, last), rank_(rank)
      { }

      template <typename ContiguousIterator>
      ranked_buffer(
        ContiguousIterator const first, ContiguousIterator const last,
        ::yampi::datatype const datatype, ::yampi::rank const rank)
        : buffer_(first, last, datatype), rank_(rank)
      { }

      ranked_buffer(
        ::yampi::buffer<T>& buffer, ::yampi::rank const rank)
        : buffer_(buffer), rank_(rank)
      { }

      ranked_buffer(
        ::yampi::buffer<T> const& buffer, ::yampi::rank const rank)
        : buffer_(buffer), rank_(rank)
      { }

      ::yampi::buffer<T> const& buffer() const { return buffer_; }
      T* data() { return buffer_.data(); }
      T const* data() const { return buffer_.data(); }
      int const& count() const { return buffer_.count(); }
      ::yampi::datatype const& datatype() const { return buffer_.datatype(); }
      ::yampi::rank const& rank() const { return rank_; }
    };


    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(T& value, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(value, rank); }

    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(T const& value, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(value, rank); }

    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(
      T& value, ::yampi::datatype const datatype, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(value, datatype, rank); }

    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(
      T const& value, ::yampi::datatype const datatype, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(value, datatype, rank); }

    template <typename ContiguousIterator>
    inline
    ::yampi::algorithm::ranked_buffer<
      typename YAMPI_remove_cv<
        typename std::iterator_traits<ContiguousIterator>::value_type>::type>
    make_ranked_buffer(
      ContiguousIterator const first, ContiguousIterator const last, ::yampi::rank const rank)
    {
      typedef
        ::yampi::algorithm::ranked_buffer<
          typename YAMPI_remove_cv<
            typename std::iterator_traits<ContiguousIterator>::value_type>::type>
        result_type;
      return result_type(first, last, rank);
    }

    template <typename ContiguousIterator>
    inline
    ::yampi::algorithm::ranked_buffer<
      typename YAMPI_remove_cv<
        typename std::iterator_traits<ContiguousIterator>::value_type>::type>
    make_ranked_buffer(
      ContiguousIterator const first, ContiguousIterator const last,
      ::yampi::datatype const datatype, ::yampi::rank const rank)
    {
      typedef
        ::yampi::algorithm::ranked_buffer<
          typename YAMPI_remove_cv<
            typename std::iterator_traits<ContiguousIterator>::value_type>::type>
        result_type;
      return result_type(first, last, datatype, rank);
    }

    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(
      ::yampi::buffer<T>& buffer, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(buffer, rank); }

    template <typename T>
    inline ::yampi::algorithm::ranked_buffer<T> make_ranked_buffer(
      ::yampi::buffer<T> const& buffer, ::yampi::rank const rank)
    { return ::yampi::algorithm::ranked_buffer<T>(buffer, rank); }
  }
}


# undef YAMPI_addressof
# undef YAMPI_remove_cv

#endif
