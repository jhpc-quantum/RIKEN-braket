#ifndef KET_MPI_GATE_PAGE_CONTROLLED_NOT_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_NOT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/detail/controlled_not_tcp.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // tcp: both of target qubit and control qubit are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_not_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::detail::controlled_not_tcp(
            mpi_policy, parallel_policy, local_state, target_qubit, control_qubit, permutation);
        }

        // tp: only target qubit is on page
        namespace controlled_not_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct controlled_not_tp
          {
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_not_tp(
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubit_mask_{control_qubit_mask},
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
              auto const one_index = zero_index bitor control_qubit_mask_;
              std::iter_swap(zero_first + one_index, one_first + one_index);
            }
          }; // struct controlled_not_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_not_detail::controlled_not_tp<StateInteger>
          make_controlled_not_tp(
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_not_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_not_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[control_qubit.qubit()]);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            mpi_policy, parallel_policy, local_state, target_qubit, permutation,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              std::iter_swap(zero_first + one_index, one_first + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            mpi_policy, parallel_policy, local_state, target_qubit, permutation,
            ::ket::mpi::gate::page::controlled_not_detail::make_controlled_not_tp(
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cp: only control qubit is on page
        namespace controlled_not_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct controlled_not_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_not_cp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor target_qubit_mask_;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            }
          }; // struct controlled_not_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::controlled_not_detail::controlled_not_cp<StateInteger>
          make_controlled_not_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace controlled_not_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& controlled_not_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not local_state.is_page_qubit(permutation[target_qubit]));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            mpi_policy, parallel_policy, local_state, control_qubit.qubit(), permutation,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            mpi_policy, parallel_policy, local_state, control_qubit.qubit(), permutation,
            ::ket::mpi::gate::page::controlled_not_detail::make_controlled_not_cp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tcp: both of target qubit and control qubit are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_controlled_not_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::controlled_not_tcp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);
        }

        // tp: only target qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_controlled_not_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::controlled_not_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);
        }

        // cp: only control qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_controlled_not_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          return ::ket::mpi::gate::page::controlled_not_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CONTROLLED_NOT_HPP
