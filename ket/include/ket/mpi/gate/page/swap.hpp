#ifndef KET_MPI_GATE_PAGE_SWAP_HPP
# define KET_MPI_GATE_PAGE_SWAP_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/swap_2p.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 2p: both of two qubits are on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& swap_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        {
          return ::ket::mpi::gate::page::detail::swap_2p(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2);
        }

        // p: only one qubit is on page
        namespace swap_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct swap_p
          {
            StateInteger nonpage_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            swap_p(
              StateInteger const nonpage_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : nonpage_qubit_mask_{nonpage_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor nonpage_qubit_mask_;
              std::iter_swap(zero_first + one_index, one_first + zero_index);
            }
          }; // struct swap_p<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::swap_detail::swap_p<StateInteger>
          make_swap_p(
            StateInteger const nonpage_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {nonpage_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace swap_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& swap_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
        {
          assert(::ket::mpi::page::is_on_page(page_permutated_qubit, local_state));
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_qubit, local_state));
          auto const nonpage_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_qubit);
          auto const nonpage_lower_bits_mask = nonpage_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            [nonpage_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_qubit_mask;
              std::iter_swap(zero_first + one_index, one_first + zero_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::swap_detail::make_swap_p(
              nonpage_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2p: both of two qubits are on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_swap_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        {
          return ::ket::mpi::gate::page::swap_2p(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2);
        }

        // p: only one qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_swap_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
        {
          return ::ket::mpi::gate::page::swap_p(
            parallel_policy, local_state, page_permutated_qubit, nonpage_permutated_qubit);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SWAP_HPP
