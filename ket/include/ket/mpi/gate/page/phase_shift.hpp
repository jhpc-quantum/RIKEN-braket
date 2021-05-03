#ifndef KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cmath>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/exp_i.hpp>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   include <ket/utility/meta/real_of.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct phase_shift_coeff
          {
            Complex phase_coefficient_;

            explicit phase_shift_coeff(Complex const& phase_coefficient)
              : phase_coefficient_{phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const, Iterator const one_first, StateInteger const index) const
            { *(one_first + index) *= phase_coefficient_; }
          }; // struct phase_shift_coeff<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift_coeff<Complex>
          make_phase_shift_coeff(Complex const& phase_coefficient)
          { return ::ket::mpi::gate::page::phase_shift_detail::phase_shift_coeff<Complex>{phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [phase_coefficient](auto const, auto const one_first, StateInteger const index)
            { *(one_first + index) *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift_coeff(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // generalized phase_shift
        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct phase_shift2
          {
            Complex modified_phase_coefficient1_;
            Complex phase_coefficient2_;

            phase_shift2(
              Complex const& modified_phase_coefficient1,
              Complex const& phase_coefficient2)
              : modified_phase_coefficient1_{modified_phase_coefficient1},
                phase_coefficient2_{phase_coefficient2}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *zero_iter -= phase_coefficient2_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient2_;
              *one_iter += zero_iter_value;
              *one_iter *= modified_phase_coefficient1_;
            }
          }; // struct phase_shift2<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift2<Complex>
          make_phase_shift2(
            Complex const& modified_phase_coefficient1, Complex const& phase_coefficient2)
          {
            return ::ket::mpi::gate::page::phase_shift_detail::phase_shift2<Complex>{
              modified_phase_coefficient1, phase_coefficient2};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift2(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient1 = one_div_root_two<Real>() * phase_coefficient1;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [modified_phase_coefficient1, phase_coefficient2](auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter -= phase_coefficient2 * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter *= phase_coefficient2;
              *one_iter += zero_iter_value;
              *one_iter *= modified_phase_coefficient1;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift2(modified_phase_coefficient1, phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct adj_phase_shift2
          {
            Complex phase_coefficient1_;
            Complex modified_phase_coefficient2_;

            adj_phase_shift2(
              Complex const& phase_coefficient1,
              Complex const& modified_phase_coefficient2)
              : phase_coefficient1_{phase_coefficient1},
                modified_phase_coefficient2_{modified_phase_coefficient2}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
              using boost::math::constants::one_div_root_two;
              *zero_iter += phase_coefficient1_ * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter *= phase_coefficient1_;
              *one_iter -= zero_iter_value;
              *one_iter *= modified_phase_coefficient2_;
            }
          }; // struct adj_phase_shift2<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift2<Complex>
          make_adj_phase_shift2(
            Complex const& phase_coefficient1, Complex const& modified_phase_coefficient2)
          {
            return ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift2<Complex>{
              phase_coefficient1, modified_phase_coefficient2};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_phase_shift2(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient1 = ::ket::utility::exp_i<complex_type>(-phase1);
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);

          using boost::math::constants::one_div_root_two;
          auto const modified_phase_coefficient2 = one_div_root_two<Real>() * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [phase_coefficient1, modified_phase_coefficient2](auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter += phase_coefficient1 * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter *= phase_coefficient1;
              *one_iter -= zero_iter_value;
              *one_iter *= modified_phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift2(phase_coefficient1, modified_phase_coefficient2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct phase_shift3
          {
            Real sine_;
            Real cosine_;
            Complex phase_coefficient2_;
            Complex sine_phase_coefficient3_;
            Complex cosine_phase_coefficient3_;

            phase_shift3(
              Real const sine, Real const cosine,
              Complex const& phase_coefficient2,
              Complex const& sine_phase_coefficient3,
              Complex const& cosine_phase_coefficient3)
              : sine_{sine},
                cosine_{cosine},
                phase_coefficient2_{phase_coefficient2},
                sine_phase_coefficient3_{sine_phase_coefficient3},
                cosine_phase_coefficient3_{cosine_phase_coefficient3}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter -= sine_phase_coefficient3_ * *one_iter;
              *one_iter *= cosine_phase_coefficient3_;
              *one_iter += sine_ * zero_iter_value;
              *one_iter *= phase_coefficient2_;
            }
          }; // struct phase_shift3<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::phase_shift3<Real, Complex>
          make_phase_shift3(
            Real const sine, Real const cosine,
            Complex const& phase_coefficient2,
            Complex const& sine_phase_coefficient3,
            Complex const& cosine_phase_coefficient3)
          {
            return ::ket::mpi::gate::page::phase_shift_detail::phase_shift3<Real, Complex>{
              sine, cosine, phase_coefficient2,
              sine_phase_coefficient3, cosine_phase_coefficient3};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift3(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(phase3);

          auto const sine_phase_coefficient3 = sine * phase_coefficient3;
          auto const cosine_phase_coefficient3 = cosine * phase_coefficient3;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [sine, cosine, phase_coefficient2,
             sine_phase_coefficient3, cosine_phase_coefficient3](
              auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine;
              *zero_iter -= sine_phase_coefficient3 * *one_iter;
              *one_iter *= cosine_phase_coefficient3;
              *one_iter += sine * zero_iter_value;
              *one_iter *= phase_coefficient2;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift3(
              sine, cosine, phase_coefficient2,
              sine_phase_coefficient3, cosine_phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Real, typename Complex>
          struct adj_phase_shift3
          {
            Real sine_;
            Real cosine_;
            Complex sine_phase_coefficient2_;
            Complex cosine_phase_coefficient2_;
            Complex phase_coefficient3_;

            adj_phase_shift3(
              Real const sine, Real const cosine,
              Complex const& sine_phase_coefficient2,
              Complex const& cosine_phase_coefficient2,
              Complex const& phase_coefficient3)
              : sine_{sine},
                cosine_{cosine},
                sine_phase_coefficient2_{sine_phase_coefficient2},
                cosine_phase_coefficient2_{cosine_phase_coefficient2},
                phase_coefficient3_{phase_coefficient3}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine_;
              *zero_iter += sine_phase_coefficient2_ * *one_iter;
              *one_iter *= cosine_phase_coefficient2_;
              *one_iter -= sine_ * zero_iter_value;
              *one_iter *= phase_coefficient3_;
            }
          }; // struct adj_phase_shift3<Real, Complex>

          template <typename Real, typename Complex>
          inline ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift3<Real, Complex>
          make_adj_phase_shift3(
            Real const sine, Real const cosine,
            Complex const& sine_phase_coefficient2,
            Complex const& cosine_phase_coefficient2,
            Complex const& phase_coefficient3)
          {
            return ::ket::mpi::gate::page::phase_shift_detail::adj_phase_shift3<Real, Complex>{
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
              phase_coefficient3};
          }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_phase_shift3(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using std::cos;
          using std::sin;
          using boost::math::constants::half;
          auto const sine = sin(half<Real>() * phase1);
          auto const cosine = cos(half<Real>() * phase1);

          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          auto const phase_coefficient2 = ::ket::utility::exp_i<complex_type>(-phase2);
          auto const phase_coefficient3 = ::ket::utility::exp_i<complex_type>(-phase3);

          auto const sine_phase_coefficient2 = sine * phase_coefficient2;
          auto const cosine_phase_coefficient2 = cosine * phase_coefficient2;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
             phase_coefficient3](
              auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              *zero_iter *= cosine;
              *zero_iter += sine_phase_coefficient2 * *one_iter;
              *one_iter *= cosine_phase_coefficient2;
              *one_iter -= sine * zero_iter_value;
              *one_iter *= phase_coefficient3;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::phase_shift_detail::make_adj_phase_shift3(
              sine, cosine, sine_phase_coefficient2, cosine_phase_coefficient2,
              phase_coefficient3));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP
