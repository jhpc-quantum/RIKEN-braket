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

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/iterator_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_remove_cv std::remove_cv
# else
#   define KET_remove_cv boost::remove_cv
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
              typename ::ket::utility::meta::iterator_of<LocalState>::type
              iterator;
            iterator const first = ::ket::utility::begin(local_state)+source_local_first_index;
            iterator const last = ::ket::utility::begin(local_state)+source_local_last_index;

            buffer.resize(source_local_last_index - source_local_first_index);
            yampi::algorithm::swap(
              yampi::ignore_status(), communicator, environment,
              yampi::make_buffer(first, last, datatype),
              yampi::make_buffer(
                ::ket::utility::begin(buffer), ::ket::utility::end(buffer), datatype),
              target_rank);
            std::copy(::ket::utility::begin(buffer), ::ket::utility::end(buffer), first);
          }
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


# undef KET_remove_cv

#endif

