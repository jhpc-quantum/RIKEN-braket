#ifndef KET_MPI_GATE_PAGE_TOFFOLI_HPP
# define KET_MPI_GATE_PAGE_TOFFOLI_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
#   define KET_DELETED_FUNCTION(function) function = delete;
# else
#   define KET_IS_DELETE function;
# endif


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& toffoli_tc0c1p(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          KET_array<
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, 2u> const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        KET_DELETED_FUNCTION((
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_tc0c1p(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          KET_array<
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, 2u> const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        ))

        KET_DELETED_FUNCTION((
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_tc0c1p(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          KET_array<
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, 2u> const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        ))
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


# undef KET_DELETED_FUNCTION

#endif
