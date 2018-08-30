#ifndef KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/page/controlled_phase_shift.hpp.old>
#else // KET_USE_DIAGONAL_LOOP
//
#ifndef KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PAGE_CONTROLLED_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cassert>
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <boost/algorithm/minmax.hpp>
# include <boost/tuple/tuple.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/iterator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // controlled_phase_shift_coeff_tcp
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_phase_shift_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 1, StateAllocator>&
        controlled_phase_shift_coeff_tcp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 1, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }


        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex>
          struct controlled_phase_shift_coeff_tcp_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;

            controlled_phase_shift_coeff_tcp_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient)
              : first_(first), phase_coefficient_(phase_coefficient)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            { *(first_+index) *= phase_coefficient_; }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline controlled_phase_shift_coeff_tcp_loop_inside<RandomAccessIterator, Complex>
          make_controlled_phase_shift_coeff_tcp_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient)
          {
            typedef
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::controlled_phase_shift_coeff_tcp_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(first, phase_coefficient);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_phase_shift_coeff_tcp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation)
        {
          static_assert(
            num_page_qubits_ >= 2,
            "num_page_qubits_ should be greater than or equal to 2");

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const permutated_target_qubit = permutation[target_qubit];
          qubit_type const permutated_cqubit = permutation[control_qubit.qubit()];
          assert(local_state.is_page_qubit(permutated_target_qubit));
          assert(local_state.is_page_qubit(permutated_cqubit));

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

          boost::tuple<qubit_type, qubit_type> const minmax_qubits
            = boost::minmax(permutated_target_qubit, permutated_cqubit);
          StateInteger const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_target_qubit - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutated_cqubit - static_cast<qubit_type>(num_nonpage_qubits));
          using boost::get;
          StateInteger const lower_bits_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                get<0u>(minmax_qubits) - static_cast<qubit_type>(num_nonpage_qubits))
              - static_cast<StateInteger>(1u);
          StateInteger const middle_bits_mask
            = (::ket::utility::integer_exp2<StateInteger>(
                 get<1u>(minmax_qubits) - static_cast<qubit_type>(1u+num_nonpage_qubits))
               - static_cast<StateInteger>(1u))
              xor lower_bits_mask;
          StateInteger const upper_bits_mask
            = compl (lower_bits_mask bitor middle_bits_mask);

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t page_id_wo_qubits = 0u;
               page_id_wo_qubits < local_state_type::num_pages/4u; ++page_id_wo_qubits)
          {
            // x0_tx0_cx
            StateInteger const base_page_id
              = ((page_id_wo_qubits bitand upper_bits_mask) << 2u)
                bitor ((page_id_wo_qubits bitand middle_bits_mask) << 1u)
                bitor (page_id_wo_qubits bitand lower_bits_mask);
            // x1_tx1_cx
            StateInteger const page_id_11
              = base_page_id bitor control_qubit_mask bitor target_qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type page_range_11 = local_state.page_range(page_id_11);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const first_11 = boost::begin(page_range_11);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(page_range_11),
              [first_11, &phase_coefficient](StateInteger const index, int const)
              { *(first_11+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(page_range_11),
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::make_controlled_phase_shift_coeff_tcp_loop_inside(
                  boost::begin(page_range_11), phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        // controlled_phase_shift_coeff_tp
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_phase_shift_coeff_tp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }


        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          struct controlled_phase_shift_coeff_tp_cl_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_phase_shift_coeff_tp_cl_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
              : first_(first),
                phase_coefficient_(phase_coefficient),
                control_qubit_mask_(control_qubit_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const zero_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const one_index = zero_index bitor control_qubit_mask_;
              *(first_+one_index) *= phase_coefficient_;
            }
          };

          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          inline controlled_phase_shift_coeff_tp_cl_loop_inside<RandomAccessIterator, Complex, StateInteger>
          make_controlled_phase_shift_coeff_tp_cl_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::controlled_phase_shift_coeff_tp_cl_loop_inside<
                  RandomAccessIterator, Complex, StateInteger>
              result_type;

            return result_type(
              first, phase_coefficient,
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask);
          }


          template <typename RandomAccessIterator, typename Complex>
          struct controlled_phase_shift_coeff_tp_cg_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;

            controlled_phase_shift_coeff_tp_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient)
              : first_(first),
                phase_coefficient_(phase_coefficient)
            { }

            void operator()(StateInteger const index, int const) const
            { *(first_+index) *= phase_coefficient_; }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline controlled_phase_shift_coeff_tp_cl_loop_inside<RandomAccessIterator, Complex>
          make_controlled_phase_shift_coeff_tp_cl_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient)
          {
            typedef
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::controlled_phase_shift_coeff_tp_cl_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(first, phase_coefficient);
          }
# endif // BOOST_NO_CXX11_LAMBDAS


          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_tp_cl(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(local_state.is_page_qubit(permutation[target_qubit]));
            assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            BitInteger const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            StateInteger const target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[target_qubit] - static_cast<qubit_type>(num_nonpage_qubits));
            StateInteger const control_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(permutation[control_qubit.qubit()]);
            StateInteger const page_lower_bits_mask = target_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const nonpage_lower_bits_mask = control_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;
            StateInteger const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

            typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
            for (std::size_t page_id_wo_qubit = 0u;
                 page_id_wo_qubit < local_state_type::num_pages/2u; ++page_id_wo_qubit)
            {
              // x0x
              StateInteger const zero_page_id
                = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                  bitor (page_id_wo_qubit bitand page_lower_bits_mask);
              // x1x
              StateInteger const one_page_id = zero_page_id bitor target_qubit_mask;

              typedef typename local_state_type::page_range_type page_range_type;
              page_range_type one_page_range = local_state.page_range(one_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const one_first = boost::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range)/2u,
                [one_first, &phase_coefficient,
                 control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  StateInteger const index_wo_qubit, int const)
                {
                  StateInteger const zero_index
                    = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                  StateInteger const one_index = zero_index bitor control_qubit_mask;
                  *(one_first+one_index) *= phase_coefficient;
                });
# else // BOOST_NO_CXX11_LAMBDAS
              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range)/2u,
                ::ket::mpi::gate::page::controlled_phase_shift_detail
                  ::make_controlled_phase_shift_coeff_tp_cl_loop_inside(
                    boost::begin(one_page_range), phase_coefficient,
                    control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
            }

            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_tp_cg(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::rank const rank)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(local_state.is_page_qubit(permutation[target_qubit]));
            assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            qubit_type const least_global_permutated_qubit(
              ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));
            BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;
            BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;

            StateInteger const mask
              = one_state_integer
                << (permutation[control_qubit.qubit()] - least_global_permutated_qubit);

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
              return local_state;

            BitInteger const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

            StateInteger const target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[target_qubit] - static_cast<qubit_type>(num_nonpage_qubits));
            StateInteger const page_lower_bits_mask = target_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;

            typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
            for (std::size_t page_id_wo_qubit = 0u;
                 page_id_wo_qubit < local_state_type::num_pages/2u; ++page_id_wo_qubit)
            {
              // x0x
              StateInteger const zero_page_id
                = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                  bitor (page_id_wo_qubit bitand page_lower_bits_mask);
              // x1x
              StateInteger const one_page_id = zero_page_id bitor target_qubit_mask;

              typedef typename local_state_type::page_range_type page_range_type;
              page_range_type one_page_range = local_state.page_range(one_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const one_first = boost::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range),
                [one_first, &phase_coefficient](StateInteger const index, int const)
                { *(one_first+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range)/2u,
                ::ket::mpi::gate::page::controlled_phase_shift_detail
                  ::make_controlled_phase_shift_coeff_tp_cg_loop_inside(
                    boost::begin(one_page_range), phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS
            }

            return local_state;
          }
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_phase_shift_coeff_tp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          static_assert(num_page_qubits_ >= 1, "num_page_qubits_ should be greater than or equal to 1");
          assert(local_state.is_page_qubit(permutation[target_qubit]));
          assert(not local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          qubit_type const least_permutated_global_qubit(
            ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));

          if (permutation[control_qubit.qubit()] < least_permutated_global_qubit)
            return ::ket::mpi::gate::page::controlled_phase_shift_detail
              ::controlled_phase_shift_coeff_tp_cl(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::controlled_phase_shift_detail
            ::controlled_phase_shift_coeff_tp_cg(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation, rank);
        }


        // controlled_phase_shift_coeff_cp
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& controlled_phase_shift_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>&
        controlled_phase_shift_coeff_cp(
          MpiPolicy const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&,
          yampi::rank const)
        { return local_state; }


        namespace controlled_phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          struct controlled_phase_shift_coeff_cp_tl_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            controlled_phase_shift_coeff_cp_tl_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
              : first_(first),
                phase_coefficient_(phase_coefficient),
                target_qubit_mask_(target_qubit_mask),
                nonpage_lower_bits_mask_(nonpage_lower_bits_mask),
                nonpage_upper_bits_mask_(nonpage_upper_bits_mask)
            { }

            void operator()(StateInteger const index_wo_qubit, int const) const
            {
              StateInteger const zero_index
                = ((index_wo_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_qubit bitand nonpage_lower_bits_mask_);
              StateInteger const one_index = zero_index bitor target_qubit_mask_;
              *(first_+one_index) *= phase_coefficient_;
            }
          };

          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          inline controlled_phase_shift_coeff_cp_tl_loop_inside<
            RandomAccessIterator, Complex, StateInteger>
          make_controlled_phase_shift_coeff_cp_tl_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          {
            typedef
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::controlled_phase_shift_coeff_cp_tl_loop_inside<
                  RandomAccessIterator, Complex, StateInteger>
              result_type;

            return result_type(
              first, phase_coefficient,
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask);
          }


          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          struct controlled_phase_shift_coeff_cp_tg_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;

            controlled_phase_shift_coeff_cp_tg_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient)
              : first_(first),
                phase_coefficient_(phase_coefficient)
            { }

            void operator()(StateInteger const index, int const) const
            { *(first_+index) *= phase_coefficient_; }
          };

          template <typename RandomAccessIterator, typename Complex, typename StateInteger>
          inline controlled_phase_shift_coeff_cp_tg_loop_inside<
            RandomAccessIterator, Complex, StateInteger>
          make_controlled_phase_shift_coeff_cp_tg_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient)
          {
            typedef
              ::ket::mpi::gate::page::controlled_phase_shift_detail
                ::controlled_phase_shift_coeff_cp_tg_loop_inside<
                  RandomAccessIterator, Complex, StateInteger>
              result_type;

            return result_type(first, phase_coefficient);
          }
# endif // BOOST_NO_CXX11_LAMBDAS


          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_cp_tl(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(not local_state.is_page_qubit(permutation[target_qubit]));
            assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            BitInteger const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

            StateInteger const target_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(permutation[target_qubit]);
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            StateInteger const control_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[control_qubit.qubit()]
                  - static_cast<qubit_type>(num_nonpage_qubits));
            StateInteger const page_lower_bits_mask
              = control_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const nonpage_lower_bits_mask
              = target_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;
            StateInteger const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

            typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
            for (std::size_t page_id_wo_qubit = 0u;
                 page_id_wo_qubit < local_state_type::num_pages/2u; ++page_id_wo_qubit)
            {
              // x0x
              StateInteger const zero_page_id
                = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                  bitor (page_id_wo_qubit bitand page_lower_bits_mask);
              // x1x
              StateInteger const one_page_id = zero_page_id bitor control_qubit_mask;

              typedef typename local_state_type::page_range_type page_range_type;
              page_range_type one_page_range = local_state.page_range(one_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const one_first = boost::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range)/2u,
                [one_first, &phase_coefficient,
                 target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
                  StateInteger const index_wo_qubit, int const)
                {
                  StateInteger const zero_index
                    = ((index_wo_qubit bitand nonpage_upper_bits_mask) << 1u)
                      bitor (index_wo_qubit bitand nonpage_lower_bits_mask);
                  StateInteger const one_index = zero_index bitor target_qubit_mask;
                  *(one_first+one_index) *= phase_coefficient;
                });
# else // BOOST_NO_CXX11_LAMBDAS
              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range)/2u,
                ::ket::mpi::gate::page::controlled_phase_shift_detail
                  ::make_controlled_phase_shift_coeff_cp_tl_loop_inside(
                  boost::begin(one_page_range), phase_coefficient,
                  target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
            }

            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename StateAllocator,
            typename StateInteger, typename BitInteger, typename PermutationAllocator>
          inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
          controlled_phase_shift_coeff_cp_tg(
            MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ::ket::mpi::qubit_permutation<
              StateInteger, BitInteger, PermutationAllocator> const& permutation,
            yampi::rank const rank)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(not local_state.is_page_qubit(permutation[target_qubit]));
            assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            qubit_type const least_global_permutated_qubit(
              ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));
            BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;
            BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;

            StateInteger const mask
              = one_state_integer
                << (permutation[target_qubit] - least_global_permutated_qubit);

            if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
              return local_state;

            BitInteger const num_nonpage_qubits
              = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);

            StateInteger const control_qubit_mask
              = ::ket::utility::integer_exp2<StateInteger>(
                  permutation[control_qubit.qubit()] - static_cast<qubit_type>(num_nonpage_qubits));
            StateInteger const page_lower_bits_mask = control_qubit_mask-static_cast<StateInteger>(1u);
            StateInteger const page_upper_bits_mask = compl page_lower_bits_mask;

            typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
            for (std::size_t page_id_wo_qubit = 0u;
                 page_id_wo_qubit < local_state_type::num_pages/2u; ++page_id_wo_qubit)
            {
              // x0x
              StateInteger const zero_page_id
                = ((page_id_wo_qubit bitand page_upper_bits_mask) << 1u)
                  bitor (page_id_wo_qubit bitand page_lower_bits_mask);
              // x1x
              StateInteger const one_page_id = zero_page_id bitor control_qubit_mask;

              typedef typename local_state_type::page_range_type page_range_type;
              page_range_type one_page_range = local_state.page_range(one_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
              typedef typename boost::range_iterator<page_range_type>::type page_iterator;
              page_iterator const one_first = boost::begin(one_page_range);

              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range),
                [one_first, &phase_coefficient](StateInteger const index, int const)
                { *(one_first+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
              using ::ket::utility::loop_n;
              loop_n(
                parallel_policy,
                boost::size(one_page_range),
                ::ket::mpi::gate::page::controlled_phase_shift_detail
                  ::make_controlled_phase_shift_coeff_cp_tg_loop_inside(
                  boost::begin(one_page_range), phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS
            }

            return local_state;
          }
        } // namespace controlled_phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>&
        controlled_phase_shift_coeff_cp(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const& permutation,
          yampi::rank const rank)
        {
          static_assert(
            num_page_qubits_ >= 1,
            "num_page_qubits_ should be greater than or equal to 1");
          assert(not local_state.is_page_qubit(permutation[target_qubit]));
          assert(local_state.is_page_qubit(permutation[control_qubit.qubit()]));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          qubit_type const least_permutated_global_qubit(
            ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));

          if (permutation[target_qubit] < least_permutated_global_qubit)
            return ::ket::mpi::gate::page::controlled_phase_shift_detail
              ::controlled_phase_shift_coeff_cp_tl(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::controlled_phase_shift_detail
            ::controlled_phase_shift_coeff_cp_tg(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation, rank);
        }
      } // namespace page
    } // namespage gate
  } // namespace mpi
} // namespace ket


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif
//
#endif // KET_USE_DIAGONAL_LOOP

