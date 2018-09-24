#ifndef KET_MPI_UTILITY_DETAIL_SWAP_LOCAL_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_SWAP_LOCAL_QUBITS_HPP

# include <boost/config.hpp>

# include <algorithm>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_cv.hpp>
# endif

# include <boost/tuple/tuple.hpp>
# include <boost/algorithm/minmax.hpp>

# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
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
      namespace swap_local_qubits_detail
      {
# ifdef BOOST_NO_CXX11_LAMBDAS
        template <
          typename LocalStateIterator, typename Qubit,
          typename StateInteger>
        struct swap_local_qubits_loop_inside
        {
          LocalStateIterator local_state_first_;
          Qubit min_qubit_;
          StateInteger min_qubit_mask_;
          StateInteger max_qubit_mask_;
          StateInteger middle_bits_mask_;

          swap_local_qubits_loop_inside(
            LocalStateIterator const local_state_first,
            Qubit const min_qubit,
            StateInteger const min_qubit_mask,
            StateInteger const max_qubit_mask,
            StateInteger const middle_bits_mask)
            : local_state_first_(local_state_first),
              min_qubit_(min_qubit),
              min_qubit_mask_(min_qubit_mask),
              max_qubit_mask_(max_qubit_mask),
              middle_bits_mask_(middle_bits_mask)
          { }

          // xxx|xxx|
          void operator()(StateInteger const value_wo_qubits, int const) const
          {
            // xxx0xxx0000
            StateInteger const base_index
              = ((value_wo_qubits bitand middle_bits_mask_)
                 << (min_qubit_+static_cast<Qubit>(1u)))
                bitor ((value_wo_qubits bitand compl middle_bits_mask_)
                       << (min_qubit_+static_cast<Qubit>(2u)));
            // xxx1xxx0000
            StateInteger const index1
              = base_index bitor max_qubit_mask_;
            // xxx0xxx1000
            StateInteger const index2
              = base_index bitor min_qubit_mask_;

            std::swap_ranges(
              local_state_first_+index1,
              local_state_first_+(index1 bitor min_qubit_mask_),
              local_state_first_+index2);
          }
        };

        template <
          typename LocalStateIterator, typename Qubit,
          typename StateInteger>
        inline swap_local_qubits_loop_inside<
          LocalStateIterator, Qubit, StateInteger>
        make_swap_local_qubits_loop_inside(
          LocalStateIterator const local_state_first,
          Qubit const min_qubit,
          StateInteger const min_qubit_mask,
          StateInteger const max_qubit_mask,
          StateInteger const middle_bits_mask)
        {
          return ket::mpi::utility::swap_local_qubits_detail::swap_local_qubits_loop_inside<
            LocalStateIterator, Qubit, StateInteger>(
              local_state_first, min_qubit,
              min_qubit_mask, max_qubit_mask, middle_bits_mask);
        }
# endif // BOOST_NO_CXX11_LAMBDAS
      }

      namespace dispatch
      {
        template <typename LoalState_>
        struct swap_local_qubits
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
            ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
          {
            typedef ket::qubit<StateInteger, BitInteger> qubit_type;
            boost::tuple<qubit_type, qubit_type> const minmax_qubits
              = boost::minmax(permutated_qubit1, permutated_qubit2);
            // 00000001000
            StateInteger const min_qubit_mask
              = ket::utility::integer_exp2<StateInteger>(boost::get<0u>(minmax_qubits));
            // 00010000000
            StateInteger const max_qubit_mask
              = ket::utility::integer_exp2<StateInteger>(boost::get<1u>(minmax_qubits));
            // 000|111|
            StateInteger const middle_bits_mask
              = ket::utility::integer_exp2<StateInteger>(
                  boost::get<1u>(minmax_qubits)-boost::get<0u>(minmax_qubits))
                - static_cast<StateInteger>(1u);

            using ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            typename ::ket::utility::meta::iterator_of<LocalState>::type const local_state_first
              = ::ket::utility::begin(local_state);
            loop_n(
              parallel_policy,
              (static_cast<StateInteger>(boost::size(local_state))
               >> boost::get<0u>(minmax_qubits)) >> 2u,
              [local_state_first, &minmax_qubits,
               min_qubit_mask, max_qubit_mask, middle_bits_mask](
                // xxx|xxx|
                StateInteger const value_wo_qubits, int const)
              {
                typedef ket::qubit<StateInteger, BitInteger> qubit_type;
                // xxx0xxx0000
                StateInteger const base_index
                  = ((value_wo_qubits bitand middle_bits_mask)
                     << (boost::get<0u>(minmax_qubits)+static_cast<qubit_type>(1u)))
                    bitor ((value_wo_qubits bitand compl middle_bits_mask)
                           << (boost::get<0u>(minmax_qubits)+static_cast<qubit_type>(2u)));
                // xxx1xxx0000
                StateInteger const index1
                  = base_index bitor max_qubit_mask;
                // xxx0xxx1000
                StateInteger const index2
                  = base_index bitor min_qubit_mask;

                std::swap_ranges(
                  local_state_first+index1,
                  local_state_first+(index1 bitor min_qubit_mask),
                  local_state_first+index2);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              (static_cast<StateInteger>(boost::size(local_state))
               >> boost::get<0u>(minmax_qubits)) >> 2u,
              ket::mpi::utility::swap_local_qubits_detail::make_swap_local_qubits_loop_inside(
                boost::begin(local_state), boost::get<0u>(minmax_qubits),
                min_qubit_mask, max_qubit_mask, middle_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
          }
        };
      }

      namespace detail
      {
        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger>
        inline void swap_local_qubits(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ket::qubit<StateInteger, BitInteger> const permutated_qubit1,
          ket::qubit<StateInteger, BitInteger> const permutated_qubit2)
        {
          typedef
            ::ket::mpi::utility::dispatch::swap_local_qubits<
              typename KET_remove_cv<LocalState>::type>
            swap_local_qubits_;
          swap_local_qubits_::call(
            parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
        }
      }
    }
  }
}


# undef KET_remove_cv

#endif

