#ifndef KET_MPI_GATE_PAGE_PAULI_X_HPP
# define KET_MPI_GATE_PAGE_PAULI_X_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>

# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/iterator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>


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
        inline RandomAccessRange& pauli_x(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& pauli_x(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace pauli_x_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator>
          struct pauli_x_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;

            pauli_x_loop_inside(
              RandomAccessIterator const zero_first,
              RandomAccessIterator const one_first)
              : zero_first_(zero_first), one_first_(one_first)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            { std::iter_swap(zero_first_+index, one_first_+index); }
          };

          template <typename RandomAccessIterator>
          inline pauli_x_loop_inside<RandomAccessIterator>
          make_pauli_x_loop_inside(
            RandomAccessIterator const zero_first,
            RandomAccessIterator const one_first)
          {
            typedef
              ::ket::mpi::gate::page::pauli_x_detail::pauli_x_loop_inside<RandomAccessIterator>
              result_type;

            return result_type(zero_first, one_first);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& pauli_x(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const lower_bits_mask = qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const upper_bits_mask = compl lower_bits_mask;

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t base_page_id = 0u;
               base_page_id < local_state_type::num_pages/2u; ++base_page_id)
          {
            // x0x
            StateInteger const zero_page_id
              = ((base_page_id bitand upper_bits_mask) << 1u)
                bitor (base_page_id bitand lower_bits_mask);
            // x1x
            StateInteger const one_page_id = zero_page_id bitor qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type zero_page_range
              = local_state.page_range(zero_page_id);
            page_range_type one_page_range
              = local_state.page_range(one_page_id);
            assert(boost::size(zero_page_range) == boost::size(one_page_range));

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [&zero_page_range, &one_page_range](StateInteger const index, int const)
              {
                std::iter_swap(
                  boost::begin(zero_page_range)+index,
                  boost::begin(one_page_range)+index);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::pauli_x_detail::make_pauli_x_loop_inside(
                boost::begin(zero_page_range), boost::begin(one_page_range)));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& conj_pauli_x(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::pauli_x(
            mpi_policy, parallel_policy, local_state, qubit, permutation);
        }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_pauli_x(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::conj_pauli_x(
            mpi_policy, parallel_policy, local_state, qubit, permutation);
        }
      }
    }
  }
}


#endif
