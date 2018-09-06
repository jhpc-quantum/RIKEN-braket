#ifndef KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP

# include <boost/config.hpp>

# include <cassert>
# include <vector>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
# endif
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/range/value_type.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/algorithm/copy.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_remove_cv std::remove_cv
# else
#   define KET_remove_cv boost::remove_cv
# endif

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
# endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct interchange_qubits
        {
          template <typename LocalState, typename Allocator, typename StateInteger>
          static void call(
            LocalState& local_state,
            std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::datatype const datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(source_local_last_index >= source_local_first_index);

            typedef
              typename boost::range_iterator<LocalState>::type
              iterator;
            iterator const first = boost::begin(local_state)+source_local_first_index;
            iterator const last = boost::begin(local_state)+source_local_last_index;

            buffer.resize(source_local_last_index - source_local_first_index);
            yampi::algorithm::swap(
              yampi::ignore_status(), communicator, environment,
              yampi::make_buffer(first, last, datatype),
              yampi::make_buffer(
                boost::begin(buffer), boost::end(buffer), datatype),
              target_rank);
            boost::copy(buffer, first);
          }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
          template <
            typename Complex, typename Allocator1, typename Allocator2, typename StateInteger>
          static void call(
            std::vector<Complex, Allocator1>& local_state,
            std::vector<Complex, Allocator2>& buffer,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::datatype const datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(source_local_last_index >= source_local_first_index);

            typedef
              typename std::vector<Complex, Allocator1>::pointer
              pointer;
            pointer const first = KET_addressof(local_state.front())+source_local_first_index;
            pointer const last = KET_addressof(local_state.front())+source_local_last_index;

            buffer.resize(source_local_last_index - source_local_first_index);
            yampi::algorithm::swap(
              yampi::ignore_status(), communicator, environment,
              yampi::make_buffer(first, last, datatype),
              yampi::make_buffer(
                KET_addressof(buffer.front()),
                KET_addressof(buffer.front())+buffer.size(),
                datatype),
              target_rank);
            std::copy(
              KET_addressof(buffer.front()), KET_addressof(buffer.front())+buffer.size(), first);
          }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
        };
      } // namespace dispatch

      namespace detail
      {
        template <typename LocalState, typename Allocator, typename StateInteger>
        inline void interchange_qubits(
          LocalState& local_state,
          std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype const datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          typedef
            ::ket::mpi::utility::dispatch::interchange_qubits<
              typename KET_remove_cv<LocalState>::type>
            interchange_qubits_;
          interchange_qubits_::call(
            local_state, buffer, source_local_first_index, source_local_last_index,
            datatype, target_rank, communicator, environment);
        }
      } // namespace detail
    }
  }
}


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_remove_cv

#endif

