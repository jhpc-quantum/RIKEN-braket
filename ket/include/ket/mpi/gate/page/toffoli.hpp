#ifndef KET_MPI_GATE_PAGE_TOFFOLI_HPP
# define KET_MPI_GATE_PAGE_TOFFOLI_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/detail/toffoli_tccp.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // tccp: all of target qubit and two control qubits are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& toffoli_tccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::detail::toffoli_tccp(
            mpi_policy, parallel_policy,
            local_state, target_qubit, control_qubit1, control_qubit2, permutation);
        }

        // tcp: target qubit and one of control qubits are on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_tcp
          {
            StateInteger nonpage_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tcp(
              StateInteger const nonpage_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : nonpage_control_qubit_mask_{nonpage_control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index_wo_nonpage_qubit) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor nonpage_control_qubit_mask_;
              std::iter_swap(first_10 + one_index, first_11 + one_index);
            }
          }; // struct toffoli_tcp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_tcp<StateInteger>
          make_toffoli_tcp(
            StateInteger const nonpage_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {nonpage_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& toffoli_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not local_state.is_page_qubit(permutation[nonpage_control_qubit.qubit()]));
          auto const nonpage_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[nonpage_control_qubit.qubit()]);
          auto const nonpage_lower_bits_mask = nonpage_control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            mpi_policy, parallel_policy, local_state,
            target_qubit, page_control_qubit, permutation,
            [nonpage_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const, auto const first_10, auto const first_11,
              StateInteger const index_wo_nonpage_qubit)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_control_qubit_mask;
              std::iter_swap(first_10 + one_index, first_11 + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            mpi_policy, parallel_policy, local_state,
            target_qubit, page_control_qubit, permutation,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tcp(
              nonpage_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // ccp: two control qubits are on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_ccp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_ccp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const, Iterator const, Iterator const first_11,
              StateInteger const index_wo_nonpage_qubit) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor target_qubit_mask_;
              std::iter_swap(first_11 + zero_index, first_11 + one_index);
            }
          }; // struct toffoli_ccp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_ccp<StateInteger>
          make_toffoli_ccp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& toffoli_ccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          assert(not local_state.is_page_qubit(permutation[target_qubit]));
          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            mpi_policy, parallel_policy, local_state,
            control_qubit1, control_qubit2, permutation,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const, auto const, auto const first_11,
              StateInteger const index_wo_nonpage_qubit)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              std::iter_swap(first_11 + zero_index, first_11 + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            mpi_policy, parallel_policy, local_state,
            control_qubit1, control_qubit2, permutation,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_ccp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tp: only target qubit is on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_tp
          {
            StateInteger control_qubits_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tp(
              StateInteger const control_qubits_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_middle_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubits_mask_{control_qubits_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_middle_bits_mask_{nonpage_middle_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubits) const
            {
              auto const index_00
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask_);
              auto const index_11 = index_00 bitor control_qubits_mask_;
              std::iter_swap(zero_first + index_11, one_first + index_11);
            }
          }; // struct toffoli_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_tp<StateInteger>
          make_toffoli_tp(
            StateInteger const control_qubits_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_middle_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              control_qubits_mask, nonpage_lower_bits_mask,
              nonpage_middle_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& toffoli_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          auto const permutated_control_qubit1 = permutation[control_qubit1.qubit()];
          auto const permutated_control_qubit2 = permutation[control_qubit2.qubit()];
          assert(not local_state.is_page_qubit(permutated_control_qubit1));
          assert(not local_state.is_page_qubit(permutated_control_qubit2));

          auto const control_qubits_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit1)
              bitor ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit2);

          auto const minmax_nonpage_permutated_qubits
            = std::minmax(permutated_control_qubit1, permutated_control_qubit2);
          auto const nonpage_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_qubits.first)
              - StateInteger{1u};
          auto const nonpage_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_qubits.second - BitInteger{1u})
               - StateInteger{1u})
              xor nonpage_lower_bits_mask;
          auto const nonpage_upper_bits_mask
            = compl (nonpage_lower_bits_mask bitor nonpage_middle_bits_mask);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            mpi_policy, parallel_policy, local_state, target_qubit, permutation,
            [control_qubits_mask, nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first,
              StateInteger const index_wo_nonpage_qubits)
            {
              auto const index_00
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask);
              auto const index_11 = index_00 bitor control_qubits_mask;
              std::iter_swap(zero_first + index_11, one_first + index_11);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            mpi_policy, parallel_policy, local_state, target_qubit, permutation,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tp(
              control_qubits_mask, nonpage_lower_bits_mask,
              nonpage_middle_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cp: only one of control qubits is on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_cp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_middle_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_control_qubit_mask_{nonpage_control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_middle_bits_mask_{nonpage_middle_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubits) const
            {
              auto const base_index
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask_);
              auto const zero_index = base_index bitor nonpage_control_qubit_mask_;
              auto const one_index = zero_index bitor target_qubit_mask_;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            }
          }; // struct toffoli_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_cp<StateInteger>
          make_toffoli_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_middle_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              target_qubit_mask, nonpage_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange& toffoli_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_nonpage_control_qubit = permutation[nonpage_control_qubit.qubit()];
          assert(not local_state.is_page_qubit(permutated_target_qubit));
          assert(not local_state.is_page_qubit(permutated_nonpage_control_qubit));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_nonpage_control_qubit);

          auto const minmax_nonpage_permutated_qubits
            = std::minmax(permutated_target_qubit, permutated_nonpage_control_qubit);
          auto const nonpage_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_qubits.first)
              - StateInteger{1u};
          auto const nonpage_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_qubits.second - BitInteger{1u})
               - StateInteger{1u})
              xor nonpage_lower_bits_mask;
          auto const nonpage_upper_bits_mask
            = compl (nonpage_lower_bits_mask bitor nonpage_middle_bits_mask);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            mpi_policy, parallel_policy, local_state, page_control_qubit.qubit(), permutation,
            [target_qubit_mask, nonpage_control_qubit_mask,
             nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubits)
            {
              auto const base_index
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask);
              auto const zero_index = base_index bitor nonpage_control_qubit_mask;
              auto const one_index = zero_index bitor target_qubit_mask;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            mpi_policy, parallel_policy, local_state, page_control_qubit.qubit(), permutation,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_cp(
              target_qubit_mask, nonpage_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tccp: all of target qubit and two control qubits are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange&
        adj_toffoli_tccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::toffoli_tccp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }

        // tcp: target qubit and one of two control qubits are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange&
        adj_toffoli_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::toffoli_tcp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, page_control_qubit, nonpage_control_qubit, permutation);
        }

        // ccp: two control qubits are on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange&
        adj_toffoli_ccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::toffoli_ccp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }

        // tp: only target qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange&
        adj_toffoli_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::toffoli_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }

        // cp: only one of control qubit is on page
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline RandomAccessRange&
        adj_toffoli_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          return ::ket::mpi::gate::page::toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, page_control_qubit, nonpage_control_qubit, permutation);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_TOFFOLI_HPP
