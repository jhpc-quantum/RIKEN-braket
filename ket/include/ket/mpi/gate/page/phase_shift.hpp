#ifndef KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <iterator>

# include <boost/math/constants/constants.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/iterator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
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
        // phase_shift_coeff
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& phase_shift_coeff(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex>
          struct phase_shift_coeff_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;

            phase_shift_coeff_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient)
              : first_(first), phase_coefficient_(phase_coefficient)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            { *(first_+index) *= phase_coefficient_; }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline phase_shift_coeff_loop_inside<RandomAccessIterator, Complex>
          make_phase_shift_coeff_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::phase_shift_coeff_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(first, phase_coefficient);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& phase_shift_coeff(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
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
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
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
            page_range_type one_page_range
              = local_state.page_range(one_page_id);

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range),
              [one_first, phase_coefficient](StateInteger const index, int const)
              { *(one_first+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift_coeff_loop_inside(
                boost::begin(one_page_range), phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        // generalized phase_shift
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift2(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& phase_shift2(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex>
          struct phase_shift2_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Complex modified_phase_coefficient1_;
            Complex phase_coefficient2_;

            phase_shift2_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2)
              : zero_first_(zero_first),
                one_first_(one_first),
                modified_phase_coefficient1_(modified_phase_coefficient1),
                phase_coefficient2_(phase_coefficient2)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              RandomAccessIterator const zero_iter = zero_first_ + index;
              RandomAccessIterator const one_iter = one_first_ + index;
              Complex const zero_iter_value = *zero_iter;

              typedef
                typename ::ket::utility::meta::real_of<Complex>::type real_type;
              using boost::math::constants::one_div_root_two;
              *zero_iter -= phase_coefficient2_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient2_;
              *one_iter += zero_iter_value;
              *one_iter *= modified_phase_coefficient1_;
            }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline phase_shift2_loop_inside<RandomAccessIterator, Complex>
          make_phase_shift2_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::phase_shift2_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(zero_first, one_first, modified_phase_coefficient1, phase_coefficient2);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& phase_shift2(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          Complex const phase_coefficient1 = ::ket::utility::exp_i<Complex>(phase1);
          Complex const phase_coefficient2 = ::ket::utility::exp_i<Complex>(phase2);

          using boost::math::constants::one_div_root_two;
          Complex const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
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
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            page_range_type zero_page_range
              = local_state.page_range(zero_page_id);
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(local_state.page_range(one_page_id));

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, modified_phase_coefficient1, phase_coefficient2](
                StateInteger const index, int const)
              {
                iterator const zero_iter = zero_first + index;
                iterator const one_iter = one_first + index;
                Complex const zero_iter_value = *zero_iter;

                *zero_iter -= phase_coefficient2 * *one_iter;
                *zero_iter *= one_div_root_two<Real>();
                *one_iter *= phase_coefficient2;
                *one_iter += zero_iter_value;
                *one_iter *= modified_phase_coefficient1;
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift2_loop_inside(
                zero_first, one_first, modified_phase_coefficient1, phase_coefficient2));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_phase_shift2(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& adj_phase_shift2(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex>
          struct adj_phase_shift2_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Complex phase_coefficient1_;
            Complex modified_phase_coefficient2_;

            adj_phase_shift2_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2)
              : zero_first_(zero_first),
                one_first_(one_first),
                phase_coefficient1_(phase_coefficient1),
                modified_phase_coefficient2_(modified_phase_coefficient2)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              RandomAccessIterator const zero_iter = zero_first_ + index;
              RandomAccessIterator const one_iter = one_first_ + index;
              Complex const zero_iter_value = *zero_iter;

              typedef
                typename ::ket::utility::meta::real_of<Complex>::type real_type;
              using boost::math::constants::one_div_root_two;
              *zero_iter += phase_coefficient1_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient1_;
              *one_iter -= zero_iter_value;
              *one_iter *= modified_phase_coefficient2_;
            }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline adj_phase_shift2_loop_inside<RandomAccessIterator, Complex>
          make_adj_phase_shift2_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::adj_phase_shift2_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(zero_first, one_first, phase_coefficient1, modified_phase_coefficient2);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& adj_phase_shift2(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          Complex const phase_coefficient1 = ::ket::utility::exp_i<Complex>(-phase1);
          Complex const phase_coefficient2 = ::ket::utility::exp_i<Complex>(-phase2);

          using boost::math::constants::one_div_root_two;
          Complex const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
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
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            page_range_type zero_page_range
              = local_state.page_range(zero_page_id);
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(local_state.page_range(one_page_id));

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, phase_coefficient1, modified_phase_coefficient2](
                StateInteger const index, int const)
              {
                iterator const zero_iter = zero_first + index;
                iterator const one_iter = one_first + index;
                Complex const zero_iter_value = *zero_iter;

                *zero_iter += phase_coefficient1 * *one_iter;
                *zero_iter *= one_div_root_two<Real>();
                *one_iter *= phase_coefficient1;
                *one_iter -= zero_iter_value;
                *one_iter *= modified_phase_coefficient2;
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift2_loop_inside(
                zero_first, one_first, phase_coefficient1, modified_phase_coefficient2));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift3(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Real const, Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& phase_shift3(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Real const, Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Real, typename Complex>
          struct phase_shift3_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Real sine_;
            Real cosine_;
            Complex phase_coefficient2_;
            Complex sine_phase_coefficient3_;
            Complex cosine_phase_coefficient3_;

            phase_shift3_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              Real const sine, Real const cosine, Complex const& phase_coefficient2,
              Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3)
              : zero_first_(zero_first),
                one_first_(one_first),
                sine_(sine),
                cosine_(cosine),
                phase_coefficient2_(phase_coefficient2),
                sine_phase_coefficient3_(sine_phase_coefficient3),
                cosine_phase_coefficient3_(cosine_phase_coefficient3)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              RandomAccessIterator const zero_iter = zero_first_ + index;
              RandomAccessIterator const one_iter = one_first_ + index;
              Complex const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter -= sine_phase_coefficient3_ * *one_iter;
              *one_iter *= cosine_phase_coefficient3_;
              *one_iter += sine_ * zero_iter_value;
              *one_iter *= phase_coefficient2_;
            }
          };

          template <typename RandomAccessIterator, typename Real, typename Complex>
          inline phase_shift3_loop_inside<RandomAccessIterator, Real, Complex>
          make_phase_shift3_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            Real const sine, Real const cosine, Complex const& phase_coefficient2,
            Complex const& sine_phase_coefficient3, Complex const& cosine_phase_coefficient3)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::phase_shift3_loop_inside<RandomAccessIterator, Real, Complex>
              result_type;

            return result_type(
              zero_first, one_first, sine, cosine, phase_coefficient2,
              sine_phase_coefficient3, cosine_phase_coefficient3);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& phase_shift3(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          Real const sine = sin(half<Real>() * phase1);
          Real const cosine = cos(half<Real>() * phase1);

          Complex const phase_coefficient2 = ::ket::utility::exp_i<Complex>(phase2);
          Complex const phase_coefficient3 = ::ket::utility::exp_i<Complex>(phase3);

          Complex const sine_phase_coefficient3 = sine * phase_coefficient3;
          Complex const cosine_phase_coefficient3 = cosine * phase_coefficient3;

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
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
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            page_range_type zero_page_range
              = local_state.page_range(zero_page_id);
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(local_state.page_range(one_page_id));

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, sine, cosine, phase_coefficient2,
               sine_phase_coefficient3, cosine_phase_coefficient3](
                StateInteger const index, int const)
              {
                iterator const zero_iter = zero_first + index;
                iterator const one_iter = one_first + index;
                Complex const zero_iter_value = *zero_iter;

                *zero_iter *= cosine;
                *zero_iter -= sine_phase_coefficient3 * *one_iter;
                *one_iter *= cosine_phase_coefficient3;
                *one_iter += sine * zero_iter_value;
                *one_iter *= phase_coefficient2;
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift3_loop_inside(
                zero_first, one_first, sine, cosine, phase_coefficient2,
                sine_phase_coefficient3, cosine_phase_coefficient3));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }


        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_phase_shift3(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Real const, Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& adj_phase_shift3(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Real const, Real const, Real const,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Real, typename Complex>
          struct adj_phase_shift3_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Real sine_;
            Real cosine_;
            Complex sine_phase_coefficient2_;
            Complex cosine_phase_coefficient2_;
            Complex phase_coefficient3_;

            adj_phase_shift3_loop_inside(
              RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3)
              : zero_first_(zero_first),
                one_first_(one_first),
                sine_(sine),
                cosine_(cosine),
                sine_phase_coefficient2_(sine_phase_coefficient2),
                cosine_phase_coefficient2_(cosine_phase_coefficient2),
                phase_coefficient3_(phase_coefficient3)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              RandomAccessIterator const zero_iter = zero_first_ + index;
              RandomAccessIterator const one_iter = one_first_ + index;
              Complex const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter += sine_phase_coefficient2_ * *one_iter;
              *one_iter *= cosine_phase_coefficient2_;
              *one_iter -= sine_ * zero_iter_value;
              *one_iter *= phase_coefficient3_;
            }
          };

          template <typename RandomAccessIterator, typename Real, typename Complex>
          inline adj_phase_shift3_loop_inside<RandomAccessIterator, Real, Complex>
          make_adj_phase_shift3_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first,
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2, Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::adj_phase_shift3_loop_inside<RandomAccessIterator, Real, Complex>
              result_type;

            return result_type(
              zero_first, one_first, sine, cosine,
              sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator, typename Real,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& adj_phase_shift3(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          Real const sine = sin(half<Real>() * phase1);
          Real const cosine = cos(half<Real>() * phase1);

          Complex const phase_coefficient2 = ::ket::utility::exp_i<Complex>(-phase2);
          Complex const phase_coefficient3 = ::ket::utility::exp_i<Complex>(-phase3);

          Complex const sine_phase_coefficient2 = sine * phase_coefficient2;
          Complex const cosine_phase_coefficient2 = cosine * phase_coefficient2;

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
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
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            page_range_type zero_page_range
              = local_state.page_range(zero_page_id);
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(local_state.page_range(one_page_id));

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, sine, cosine,
               sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3](
                StateInteger const index, int const)
              {
                iterator const zero_iter = zero_first + index;
                iterator const one_iter = one_first + index;
                Complex const zero_iter_value = *zero_iter;

                *zero_iter *= cosine;
                *zero_iter += sine_phase_coefficient2 * *one_iter;
                *one_iter *= cosine_phase_coefficient2;
                *one_iter -= sine * zero_iter_value;
                *one_iter *= phase_coefficient3;
              });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift3_loop_inside(
                zero_first, one_first, sine, cosine,
                sine_phase_coefficient2, cosine_phase_coefficient2, phase_coefficient3));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif

