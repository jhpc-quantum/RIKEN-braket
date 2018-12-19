#ifndef KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <iterator>
# include <utility>

# include <boost/math/constants/constants.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/iterator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
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
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<
            typename boost::range_iterator<RandomAccessRange>::type>::type,
          typename ::ket::utility::meta::real_of<
            typename boost::range_iterator<RandomAccessRange>::type>::type>
        zero_one_probabilities(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<
              typename boost::range_iterator<RandomAccessRange>::type>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator> const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<Complex>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }


        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator>
          struct zero_one_probabilities_loop_inside
          {
            long double& zero_probability_;
            long double& one_probability_;
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;

            zero_one_probabilities_loop_inside(
              long double& zero_probability,
              long double& one_probability,
              RandomAccessIterator const zero_first,
              RandomAccessIterator const one_first)
              : zero_probability_(zero_probability),
                one_probability_(one_probability),
                zero_first_(zero_first),
                one_first_(one_first)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              using std::norm;
              zero_probability_ += static_cast<long double>(norm(*(zero_first_+index)));
              one_probability_ += static_cast<long double>(norm(*(one_first_+index)));
            }
          };

          template <typename RandomAccessIterator>
          inline zero_one_probabilities_loop_inside<RandomAccessIterator> make_zero_one_probabilities_loop_inside(
            long double& zero_probability, long double& one_probability,
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first)
          {
            typedef
              ::ket::mpi::gate::page::projective_measurement_detail::zero_one_probabilities_loop_inside<RandomAccessIterator>
              result_type;
            return result_type(zero_probability, one_probability, zero_first, one_first);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> const& local_state,
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

          long double zero_probability = 0.0l;
          long double one_probability = 0.0l;

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

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [&zero_probability, &one_probability, zero_first, one_first](
                StateInteger const index, int const)
              {
                using std::norm;
                zero_probability += static_cast<long double>(norm(*(zero_first+index)));
                one_probability += static_cast<long double>(norm(*(one_first+index)));
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::projective_measurement_detail::make_zero_one_probabilities_loop_inside(
                zero_probability, one_probability,
                boost::begin(zero_page_range),
                boost::begin(one_page_range)));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          typedef typename ::ket::utility::meta::real_of<Complex>::type real_type;
          return std::make_pair(
            static_cast<real_type>(zero_probability), static_cast<real_type>(one_probability));
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_zero(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }


        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Real>
          struct change_state_after_measuring_zero_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Real multiplier_;

            change_state_after_measuring_zero_loop_inside(
              RandomAccessIterator const zero_first,
              RandomAccessIterator const one_first,
              Real const multiplier)
              : zero_first_(zero_first),
                one_first_(one_first),
                multiplier_(multiplier)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              typedef
                typename std::iterator_traits<RandomAccessIterator>::value_type
                complex_type;
              *(zero_first_+index) *= multiplier_;
              *(one_first_+index) = static_cast<complex_type>(static_cast<Real>(0));
            }
          };

          template <typename RandomAccessIterator, typename Real>
          inline change_state_after_measuring_zero_loop_inside<RandomAccessIterator, Real>
          make_change_state_after_measuring_zero_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first, Real const multiplier)
          {
            typedef
              ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_zero_loop_inside<
                RandomAccessIterator, Real>
              result_type;
            return result_type(zero_first, one_first, multiplier);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const zero_probability,
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

          using std::pow;
          using boost::math::constants::half;
          Real const multiplier = pow(zero_probability, -half<Real>());

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

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, multiplier](StateInteger const index, int const)
              {
                *(zero_first+index) *= multiplier;
                *(one_first+index) = static_cast<Complex>(static_cast<Real>(0));
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::projective_measurement_detail::make_change_state_after_measuring_zero_loop_inside(
                boost::begin(zero_page_range), boost::begin(one_page_range), multiplier));
# endif // BOOST_NO_CXX11_LAMBDAS
          }
        }



        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_one(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }


        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Real>
          struct change_state_after_measuring_one_loop_inside
          {
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;
            Real multiplier_;

            change_state_after_measuring_one_loop_inside(
              RandomAccessIterator const zero_first,
              RandomAccessIterator const one_first,
              Real const multiplier)
              : zero_first_(zero_first),
                one_first_(one_first),
                multiplier_(multiplier)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              typedef
                typename std::iterator_traits<RandomAccessIterator>::value_type
                complex_type;
              *(zero_first_+index) = static_cast<complex_type>(static_cast<Real>(0));
              *(one_first_+index) *= multiplier_;
            }
          };

          template <typename RandomAccessIterator, typename Real>
          inline change_state_after_measuring_one_loop_inside<RandomAccessIterator, Real>
          make_change_state_after_measuring_one_loop_inside(
            RandomAccessIterator const zero_first, RandomAccessIterator const one_first, Real const multiplier)
          {
            typedef
              ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_one_loop_inside<
                RandomAccessIterator, Real>
              result_type;
            return result_type(zero_first, one_first, multiplier);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const one_probability,
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

          using std::pow;
          using boost::math::constants::half;
          Real const multiplier = pow(one_probability, -half<Real>());

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

# ifndef BOOST_NO_CXX11_LAMBDAS
            typedef typename boost::range_iterator<page_range_type>::type iterator;
            iterator const zero_first = boost::begin(zero_page_range);
            iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, multiplier](StateInteger const index, int const)
              {
                *(zero_first+index) = static_cast<Complex>(static_cast<Real>(0));
                *(one_first+index) *= multiplier;
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::projective_measurement_detail::make_change_state_after_measuring_one_loop_inside(
                boost::begin(zero_page_range), boost::begin(one_page_range), multiplier));
# endif // BOOST_NO_CXX11_LAMBDAS
          }
        }
      }
    }
  }
}


#endif

