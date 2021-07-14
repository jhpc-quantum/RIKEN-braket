#ifndef KET_MPI_GATE_PAGE_TOFFOLI_HPP
# define KET_MPI_GATE_PAGE_TOFFOLI_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/is_on_page.hpp>
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
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& toffoli_tccp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          return ::ket::mpi::gate::page::detail::toffoli_tccp(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }

        // tcp: target qubit and one of control qubits are on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_tcp
          {
            StateInteger nonpage_permutated_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tcp(
              StateInteger const nonpage_permutated_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : nonpage_permutated_control_qubit_mask_{nonpage_permutated_control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor nonpage_permutated_control_qubit_mask_;
              std::iter_swap(first_10 + one_index, first_11 + one_index);
            }
          }; // struct toffoli_tcp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_tcp<StateInteger>
          make_toffoli_tcp(
            StateInteger const nonpage_permutated_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {nonpage_permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& toffoli_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));
          auto const nonpage_permutated_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_control_qubit);
          auto const nonpage_lower_bits_mask = nonpage_permutated_control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            parallel_policy, local_state,
            permutated_target_qubit, page_permutated_control_qubit,
            [nonpage_permutated_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const, auto const first_10, auto const first_11,
              StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_permutated_control_qubit_mask;
              std::iter_swap(first_10 + one_index, first_11 + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            parallel_policy, local_state,
            permutated_target_qubit, page_permutated_control_qubit,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tcp(
              nonpage_permutated_control_qubit_mask,
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
            StateInteger permutated_target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_ccp(
              StateInteger const permutated_target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : permutated_target_qubit_mask_{permutated_target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const, Iterator const, Iterator const first_11,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor permutated_target_qubit_mask_;
              std::iter_swap(first_11 + zero_index, first_11 + one_index);
            }
          }; // struct toffoli_ccp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_ccp<StateInteger>
          make_toffoli_ccp(
            StateInteger const permutated_target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& toffoli_ccp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          auto const permutated_target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = permutated_target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            parallel_policy, local_state,
            permutated_control_qubit1, permutated_control_qubit2,
            [permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const, auto const, auto const first_11,
              StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor permutated_target_qubit_mask;
              std::iter_swap(first_11 + zero_index, first_11 + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<1u>(
            parallel_policy, local_state,
            permutated_control_qubit1, permutated_control_qubit2,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_ccp(
              permutated_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tp: only target qubit is on page
        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct toffoli_tp
          {
            StateInteger permutated_control_qubits_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tp(
              StateInteger const permutated_control_qubits_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_middle_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : permutated_control_qubits_mask_{permutated_control_qubits_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_middle_bits_mask_{nonpage_middle_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubits, int const) const
            {
              auto const index_00
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask_);
              auto const index_11 = index_00 bitor permutated_control_qubits_mask_;
              std::iter_swap(zero_first + index_11, one_first + index_11);
            }
          }; // struct toffoli_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_tp<StateInteger>
          make_toffoli_tp(
            StateInteger const permutated_control_qubits_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_middle_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              permutated_control_qubits_mask, nonpage_lower_bits_mask,
              nonpage_middle_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& toffoli_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state));
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state));

          auto const permutated_control_qubits_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit1)
              bitor ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit2);

          auto const minmax_nonpage_permutated_control_qubits
            = std::minmax(permutated_control_qubit1, permutated_control_qubit2);
          auto const nonpage_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_control_qubits.first)
              - StateInteger{1u};
          auto const nonpage_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(minmax_nonpage_permutated_control_qubits.second - BitInteger{1u})
               - StateInteger{1u})
              xor nonpage_lower_bits_mask;
          auto const nonpage_upper_bits_mask = compl (nonpage_lower_bits_mask bitor nonpage_middle_bits_mask);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            parallel_policy, local_state, permutated_target_qubit,
            [permutated_control_qubits_mask, nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first,
              StateInteger const index_wo_nonpage_qubits, int const)
            {
              auto const index_00
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask);
              auto const index_11 = index_00 bitor permutated_control_qubits_mask;
              std::iter_swap(zero_first + index_11, one_first + index_11);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tp(
              permutated_control_qubits_mask, nonpage_lower_bits_mask,
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
            StateInteger permutated_target_qubit_mask_;
            StateInteger nonpage_permutated_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_cp(
              StateInteger const permutated_target_qubit_mask,
              StateInteger const nonpage_permutated_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_middle_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : permutated_target_qubit_mask_{permutated_target_qubit_mask},
                nonpage_permutated_control_qubit_mask_{nonpage_permutated_control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_middle_bits_mask_{nonpage_middle_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubits, int const) const
            {
              auto const base_index
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask_);
              auto const zero_index = base_index bitor nonpage_permutated_control_qubit_mask_;
              auto const one_index = zero_index bitor permutated_target_qubit_mask_;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            }
          }; // struct toffoli_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::toffoli_detail::toffoli_cp<StateInteger>
          make_toffoli_cp(
            StateInteger const permutated_target_qubit_mask,
            StateInteger const nonpage_permutated_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_middle_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            return {
              permutated_target_qubit_mask, nonpage_permutated_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& toffoli_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_control_qubit, local_state));

          auto const permutated_target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_permutated_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_control_qubit);

          using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
          auto const minmax_nonpage_permutated_qubits
            = static_cast<std::pair<permutated_qubit_type, permutated_qubit_type>>(
                std::minmax(permutated_target_qubit, ::ket::mpi::remove_control(nonpage_permutated_control_qubit)));
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
            parallel_policy, local_state, page_permutated_control_qubit,
            [permutated_target_qubit_mask, nonpage_permutated_control_qubit_mask,
             nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubits, int const)
            {
              auto const base_index
                = ((index_wo_nonpage_qubits bitand nonpage_upper_bits_mask) << 2u)
                  bitor ((index_wo_nonpage_qubits bitand nonpage_middle_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubits bitand nonpage_lower_bits_mask);
              auto const zero_index = base_index bitor nonpage_permutated_control_qubit_mask;
              auto const one_index = zero_index bitor permutated_target_qubit_mask;
              std::iter_swap(one_first + zero_index, one_first + one_index);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<2u>(
            parallel_policy, local_state, page_permutated_control_qubit,
            ::ket::mpi::gate::page::toffoli_detail::make_toffoli_cp(
              permutated_target_qubit_mask, nonpage_permutated_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // tccp: all of target qubit and two control qubits are on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange&
        adj_toffoli_tccp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          return ::ket::mpi::gate::page::toffoli_tccp(
            parallel_policy, local_state,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }

        // tcp: target qubit and one of two control qubits are on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange&
        adj_toffoli_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit)
        {
          return ::ket::mpi::gate::page::toffoli_tcp(
            parallel_policy, local_state,
            permutated_target_qubit, page_permutated_control_qubit, nonpage_permutated_control_qubit);
        }

        // ccp: two control qubits are on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange&
        adj_toffoli_ccp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          return ::ket::mpi::gate::page::toffoli_ccp(
            parallel_policy, local_state,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }

        // tp: only target qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange&
        adj_toffoli_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2)
        {
          return ::ket::mpi::gate::page::toffoli_tp(
            parallel_policy, local_state,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }

        // cp: only one of control qubit is on page
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange&
        adj_toffoli_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const page_permutated_control_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const nonpage_permutated_control_qubit)
        {
          return ::ket::mpi::gate::page::toffoli_cp(
            parallel_policy, local_state,
            permutated_target_qubit, page_permutated_control_qubit, nonpage_permutated_control_qubit);
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_TOFFOLI_HPP
