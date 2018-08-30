#ifndef KET_MPI_GATE_PAGE_TOFFOLI_HPP
# define KET_MPI_GATE_PAGE_TOFFOLI_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <algorithm>
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/tuple/tuple.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/algorithm/sort.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
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
        inline RandomAccessRange& toffoli_tccp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_tccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_tccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_tccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        toffoli_tccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(num_page_qubits_ >= 3, "num_page_qubits_ should be greater than or equal to 3");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_target_qubit = permutation[target_qubit];
          qubit_type const permutated_cqubit1 = permutation[control_qubit1.qubit()];
          qubit_type const permutated_cqubit2 = permutation[control_qubit2.qubit()];
          assert(local_state.is_page_qubit(permutated_target_qubit));
          assert(local_state.is_page_qubit(permutated_cqubit1));
          assert(local_state.is_page_qubit(permutated_cqubit2));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          KET_array<qubit_type, 3u> sorted_permutated_qubits
            = {permutated_target_qubit, permutated_cqubit1, permutated_cqubit2};
          boost::sort(sorted_permutated_qubits);

          StateInteger const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_target_qubit - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const control_qubits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_cqubit1 - static_cast<qubit_type>(num_nonpage_qubits))
              bitor ::ket::utility::integer_exp2<StateInteger>(
                      permutated_cqubit2 - static_cast<qubit_type>(num_nonpage_qubits));

          KET_array<StateInteger, 4u> bits_mask;
          bits_mask[0u]
            = ::ket::utility::integer_exp2<StateInteger>(
                sorted_permutated_qubits[0u] - static_cast<qubit_type>(num_nonpage_qubits))
              - static_cast<StateInteger>(1u);
          bits_mask[1u]
            = (::ket::utility::integer_exp2<StateInteger>(
                 sorted_permutated_qubits[1u]-static_cast<qubit_type>(1u+num_nonpage_qubits))
               - static_cast<StateInteger>(1u))
              xor bits_mask[0u];
          bits_mask[2u]
            = (::ket::utility::integer_exp2<StateInteger>(
                 sorted_permutated_qubits[2u]-static_cast<qubit_type>(2u+num_nonpage_qubits))
               - static_cast<StateInteger>(1u))
              xor (bits_mask[0u] bitor bits_mask[1u]);
          bits_mask[3u]
            = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/8u; ++page_id_wo_qubits)
          {
            // x0_cx0_tx0_cx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand bits_mask[3u]) << 3u)
                bitor ((page_id_wo_qubits bitand bits_mask[2u]) << 2u)
                bitor ((page_id_wo_qubits bitand bits_mask[1u]) << 1u)
                bitor (page_id_wo_qubits bitand bits_mask[0u]);
            // x1_cx0_tx1_cx
            StateInteger const control_on_page_id = base_page_id bitor control_qubits_mask;
            // x1_cx1_tx1_cx
            StateInteger const target_control_on_page_id
              = control_on_page_id bitor target_qubit_mask;

            local_state.swap_pages(control_on_page_id, target_control_on_page_id);
          }

          return local_state;
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& toffoli_tcp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename StateInteger>
          struct toffoli_tcp_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            StateInteger nonpage_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tcp_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              StateInteger const nonpage_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_upper_bits_mask)
              : zero_first_(zero_first),
                one_first_(one_first),
                nonpage_control_qubit_mask_(nonpage_control_qubit_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const zero_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const one_index = zero_index bitor nonpage_control_qubit_mask_;
              std::iter_swap(zero_first_+one_index, one_first_+one_index);
            }
          }; // struct toffoli_tcp_loop_inside<RandomAccessIterator, StateInteger>

          template <typename RandomAccessIterator, typename StateInteger>
          inline toffoli_tcp_loop_inside<RandomAccessIterator, StateInteger>
          make_toffoli_tcp_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            StateInteger const nonpage_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::toffoli_detail::toffoli_tcp_loop_inside<RandomAccessIterator, StateInteger>
              result_type;

            return result_type(
              zero_first, one_first,
              nonpage_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        toffoli_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(num_page_qubits_ >= 3, "num_page_qubits_ should be greater than or equal to 3");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_target_qubit = permutation[target_qubit];
          qubit_type const permutated_page_cqubit = permutation[page_control_qubit.qubit()];
          assert(local_state.is_page_qubit(permutated_target_qubit));
          assert(local_state.is_page_qubit(permutated_page_cqubit));
          assert(not local_state.is_page_qubit(permutation[nonpage_control_qubit.qubit()]));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          boost::tuple<qubit_type, qubit_type> const minmax_page_permutated_qubits
            = boost::minmax(permutated_target_qubit, permutated_page_cqubit);

          StateInteger const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_target_qubit - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const page_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_page_cqubit - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const nonpage_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[nonpage_control_qubit.qubit()]);

          using boost::get;
          StateInteger const page_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                get<0u>(minmax_page_permutated_qubits) - static_cast<qubit_type>(num_nonpage_qubits))
              - static_cast<StateInteger>(1u);
          StateInteger const page_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(
                 get<1u>(minmax_page_permutated_qubits) - static_cast<qubit_type>(1u+num_nonpage_qubits))
               - static_cast<StateInteger>(1u))
              xor page_lower_bits_mask;
          StateInteger const page_upper_bits_mask
            = compl (page_lower_bits_mask bitor page_middle_bits_mask);
          StateInteger const nonpage_lower_bits_mask = nonpage_control_qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/4u; ++page_id_wo_qubits)
          {
            // x0_tx0_cx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand page_upper_bits_mask) << 2u)
                bitor ((page_id_wo_qubits bitand page_middle_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand page_lower_bits_mask);
            // x0_tx1_cx
            StateInteger const control_on_page_id
              = base_page_id bitor page_control_qubit_mask;
            // x1_tx1_cx
            StateInteger const target_control_on_page_id
              = control_on_page_id bitor target_qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type zero_page_range = local_state.page_range(control_on_page_id);
            page_range_type one_page_range = local_state.page_range(target_control_on_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const zero_first = boost::begin(zero_page_range);
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range)/2u,
              [zero_first, one_first,
               nonpage_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                StateInteger const zero_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                StateInteger const one_index = zero_index bitor nonpage_control_qubit_mask;
                std::iter_swap(zero_first+one_index, one_first+one_index);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range)/2u,
              ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tcp_loop_inside(
                boost::begin(zero_page_range), boost::begin(one_page_range),
                nonpage_control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& toffoli_ccp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_ccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_ccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_ccp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename StateInteger>
          struct toffoli_ccp_loop_inside
          {
            RandomAccessIterator one_first_;
            StateInteger nonpage_target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_ccp_loop_inside(
              RandomAccessIterator const one_first,
              StateInteger const nonpage_target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_upper_bits_mask)
              : one_first_(one_first),
                nonpage_target_qubit_mask_(nonpage_target_qubit_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const zero_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const one_index = zero_index bitor nonpage_target_qubit_mask_;
              std::iter_swap(one_first_+zero_index, one_first_+one_index);
            }
          }; // struct toffoli_ccp_loop_inside<RandomAccessIterator, StateInteger>

          template <typename RandomAccessIterator, typename StateInteger>
          inline toffoli_ccp_loop_inside<RandomAccessIterator, StateInteger>
          make_toffoli_ccp_loop_inside(
            RandomAccessIterator const one_first,
            StateInteger const nonpage_target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::toffoli_detail::toffoli_ccp_loop_inside<RandomAccessIterator, StateInteger>
              result_type;

            return result_type(
              one_first,
              nonpage_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        toffoli_ccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(num_page_qubits_ >= 3, "num_page_qubits_ should be greater than or equal to 3");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_cqubit1 = permutation[control_qubit1.qubit()];
          qubit_type const permutated_cqubit2 = permutation[control_qubit2.qubit()];
          assert(not local_state.is_page_qubit(permutation[target_qubit]));
          assert(local_state.is_page_qubit(permutated_cqubit1));
          assert(local_state.is_page_qubit(permutated_cqubit2));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          boost::tuple<qubit_type, qubit_type> const minmax_page_permutated_qubits
            = boost::minmax(permutated_cqubit1, permutated_cqubit2);

          StateInteger const page_control_qubits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_cqubit1 - static_cast<qubit_type>(num_nonpage_qubits))
              bitor ::ket::utility::integer_exp2<StateInteger>(
                      permutated_cqubit2 - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const nonpage_target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);

          using boost::get;
          StateInteger const page_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                get<0u>(minmax_page_permutated_qubits) - static_cast<qubit_type>(num_nonpage_qubits))
              - static_cast<StateInteger>(1u);
          StateInteger const page_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(
                 get<1u>(minmax_page_permutated_qubits) - static_cast<qubit_type>(1u+num_nonpage_qubits))
               - static_cast<StateInteger>(1u))
              xor page_lower_bits_mask;
          StateInteger const page_upper_bits_mask
            = compl (page_lower_bits_mask bitor page_middle_bits_mask);
          StateInteger const nonpage_lower_bits_mask = nonpage_target_qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/4u; ++page_id_wo_qubits)
          {
            // x0_cx0_cx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand page_upper_bits_mask) << 2u)
                bitor ((page_id_wo_qubits bitand page_middle_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand page_lower_bits_mask);
            // x1_cx1_cx
            StateInteger const control_on_page_id
              = base_page_id bitor page_control_qubits_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type one_page_range = local_state.page_range(control_on_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range)/2u,
              [one_first,
               nonpage_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                StateInteger const zero_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                StateInteger const one_index = zero_index bitor nonpage_target_qubit_mask;
                std::iter_swap(one_first+zero_index, one_first+one_index);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range)/2u,
              ::ket::mpi::gate::page::toffoli_detail::make_toffoli_ccp_loop_inside(
                boost::begin(one_page_range),
                nonpage_target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& toffoli_tp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename StateInteger>
          struct toffoli_tp_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            StateInteger control_qubits_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_tp_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              StateInteger const control_qubits_mask, StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_middle_bits_mask, StateInteger const nonpage_upper_bits_mask)
              : zero_first_(zero_first),
                one_first_(one_first),
                control_qubits_mask_(control_qubits_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_middle_bits_mask_(nonpage_middle_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const zero_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_qubit bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const one_index = zero_index bitor control_qubits_mask_;
              std::iter_swap(zero_first_+one_index, one_first_+one_index);
            }
          }; // struct toffoli_tp_loop_inside<RandomAccessIterator, StateInteger>

          template <typename RandomAccessIterator, typename StateInteger>
          inline toffoli_tp_loop_inside<RandomAccessIterator, StateInteger>
          make_toffoli_tp_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            StateInteger const control_qubits_mask, StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_middle_bits_mask, StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::toffoli_detail::toffoli_tp_loop_inside<RandomAccessIterator, StateInteger>
              result_type;

            return result_type(
              zero_first, one_first,
              control_qubits_mask, nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        toffoli_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(num_page_qubits_ >= 3, "num_page_qubits_ should be greater than or equal to 3");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_cqubit1 = permutation[control_qubit1.qubit()];
          qubit_type const permutated_cqubit2 = permutation[control_qubit2.qubit()];
          assert(local_state.is_page_qubit(permutation[target_qubit]));
          assert(not local_state.is_page_qubit(permutated_cqubit1));
          assert(not local_state.is_page_qubit(permutated_cqubit2));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          boost::tuple<qubit_type, qubit_type> const minmax_nonpage_permutated_qubits
            = boost::minmax(permutated_cqubit1, permutated_cqubit2);

          StateInteger const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[target_qubit] - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const control_qubits_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_cqubit1)
              bitor ::ket::utility::integer_exp2<StateInteger>(permutated_cqubit2);

          StateInteger const page_lower_bits_mask = target_qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;
          using boost::get;
          StateInteger const nonpage_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(get<0u>(minmax_nonpage_permutated_qubits))
              - static_cast<StateInteger>(1u);
          StateInteger const nonpage_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(get<1u>(minmax_nonpage_permutated_qubits) - static_cast<qubit_type>(1u))
               - static_cast<StateInteger>(1u))
              xor nonpage_lower_bits_mask;
          StateInteger const nonpage_upper_bits_mask
            = compl (nonpage_lower_bits_mask bitor nonpage_middle_bits_mask);

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/2u; ++page_id_wo_qubits)
          {
            // x0_tx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand page_upper_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand page_lower_bits_mask);
            // x1_tx
            StateInteger const target_on_page_id
              = base_page_id bitor target_qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type zero_page_range = local_state.page_range(base_page_id);
            page_range_type one_page_range = local_state.page_range(target_on_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const zero_first = boost::begin(zero_page_range);
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range)/4u,
              [zero_first, one_first,
               control_qubits_mask, nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                StateInteger const zero_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 2u)
                    bitor ((index_wo_qubit bitand nonpage_middle_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                StateInteger const one_index = zero_index bitor control_qubits_mask;
                std::iter_swap(zero_first+one_index, one_first+one_index);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range)/4u,
              ::ket::mpi::gate::page::toffoli_detail::make_toffoli_tp_loop_inside(
                boost::begin(zero_page_range), boost::begin(one_page_range),
                control_qubits_mask, nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& toffoli_cp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& toffoli_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>& toffoli_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 2, StateAllocator>& toffoli_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 2, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const&,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        namespace toffoli_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename StateInteger>
          struct toffoli_cp_loop_inside
          {
            RandomAccessIterator one_first_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_middle_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            toffoli_cp_loop_inside(
              RandomAccessIterator const one_first,
              StateInteger const target_qubit_mask, StateInteger const nonpage_control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_middle_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
              : one_first_(one_first),
                target_qubit_mask_(target_qubit_mask),
                nonpage_control_qubit_mask_(nonpage_control_qubit_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_middle_bits_mask_(nonpage_middle_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const base_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 2u)
                  bitor ((index_wo_qubit bitand nonpage_middle_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const zero_index = base_index bitor nonpage_control_qubit_mask_;
              StateInteger const one_index = zero_index bitor target_qubit_mask_;
              std::iter_swap(one_first_+zero_index, one_first_+one_index);
            }
          }; // struct toffoli_cp_loop_inside<RandomAccessIterator, StateInteger>

          template <typename RandomAccessIterator, typename StateInteger>
          inline toffoli_cp_loop_inside<RandomAccessIterator, StateInteger>
          make_toffoli_cp_loop_inside(
            RandomAccessIterator const one_first,
            StateInteger const target_qubit_mask, StateInteger const nonpage_control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask, StateInteger const nonpage_middle_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::toffoli_detail::toffoli_cp_loop_inside<RandomAccessIterator, StateInteger>
              result_type;

            return result_type(
              one_first,
              target_qubit_mask, nonpage_control_qubit_mask,
              nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace toffoli_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        toffoli_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& page_control_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const& nonpage_control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(num_page_qubits_ >= 3, "num_page_qubits_ should be greater than or equal to 3");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_target_qubit = permutation[target_qubit];
          qubit_type const permutated_nonpage_cqubit = permutation[nonpage_control_qubit.qubit()];
          assert(not local_state.is_page_qubit(permutated_target_qubit));
          assert(local_state.is_page_qubit(permutation[page_control_qubit.qubit()]));
          assert(not local_state.is_page_qubit(permutated_nonpage_cqubit));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          boost::tuple<qubit_type, qubit_type> const minmax_nonpage_permutated_qubits
            = boost::minmax(permutated_target_qubit, permutated_nonpage_cqubit);

          StateInteger const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          StateInteger const page_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[page_control_qubit.qubit()] - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const nonpage_control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_nonpage_cqubit);

          StateInteger const page_lower_bits_mask = page_control_qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;
          using boost::get;
          StateInteger const nonpage_lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(get<0u>(minmax_nonpage_permutated_qubits))
              - static_cast<StateInteger>(1u);
          StateInteger const nonpage_middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(get<1u>(minmax_nonpage_permutated_qubits) - static_cast<qubit_type>(1u))
               - static_cast<StateInteger>(1u))
              xor nonpage_lower_bits_mask;
          StateInteger const nonpage_upper_bits_mask
            = compl (nonpage_lower_bits_mask bitor nonpage_middle_bits_mask);

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/2u; ++page_id_wo_qubits)
          {
            // x0_cx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand page_upper_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand page_lower_bits_mask);
            // x1_cx
            StateInteger const control_on_page_id
              = base_page_id bitor page_control_qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type one_page_range = local_state.page_range(control_on_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range)/4u,
              [one_first,
               target_qubit_mask, nonpage_control_qubit_mask,
               nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask](
                StateInteger const index_wo_qubit, int const)
              {
                StateInteger const base_index
                  = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 2u)
                    bitor ((index_wo_qubit bitand nonpage_middle_bits_mask) << 1u)
                    bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                StateInteger const zero_index = base_index bitor nonpage_control_qubit_mask;
                StateInteger const one_index = zero_index bitor target_qubit_mask;
                std::iter_swap(one_first+zero_index, one_first+one_index);
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range)/4u,
              ::ket::mpi::gate::page::toffoli_detail::make_toffoli_cp_loop_inside(
                boost::begin(one_page_range),
                target_qubit_mask, nonpage_control_qubit_mask,
                nonpage_lower_bits_mask, nonpage_middle_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        adj_toffoli_tccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        adj_toffoli_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        adj_toffoli_ccp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        adj_toffoli_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        adj_toffoli_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

# undef KET_array

#endif
