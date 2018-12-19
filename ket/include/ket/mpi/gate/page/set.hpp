#ifndef KET_MPI_GATE_PAGE_SET_HPP
# define KET_MPI_GATE_PAGE_SET_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>

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
        inline RandomAccessRange& set(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& set(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace set_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename Real, typename RandomAccessIterator>
          struct set_loop1_inside
          {
            Real& one_probability_;
            RandomAccessIterator zero_first_;
            RandomAccessIterator one_first_;

            set_loop1_inside(
              Real& one_probability,
              RandomAccessIterator const zero_first,
              RandomAccessIterator const one_first)
              : one_probability_(one_probability),
                zero_first_(zero_first),
                one_first_(one_first)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            {
              typedef
                typename std::iterator_traits<RandomAccessIterator>::value_type
                complex_type;
              *(zero_first_+index) = static_cast<complex_type>(0);

              using std::norm;
              one_probability_ += norm(*(one_first_+index));
            }
          };

          template <typename Real, typename RandomAccessIterator>
          inline set_loop1_inside<Real, RandomAccessIterator> make_set_loop1_inside(
            Real& one_probability,
            RandomAccessIterator const zero_first,
            RandomAccessIterator const one_first)
          {
            typedef
              ::ket::mpi::gate::page::set_detail::set_loop1_inside<Real, RandomAccessIterator>
              result_type;

            return result_type(one_probability, zero_first, one_first);
          }


          template <typename RandomAccessIterator, typename Real>
          struct set_loop2_inside
          {
            RandomAccessIterator one_first_;
            Real multiplier_;

            set_loop2_inside(
              RandomAccessIterator const one_first,
              Real const multiplier)
              : one_first_(one_first), multiplier_(multiplier)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            { *(one_first_+index) *= multiplier_; }
          };

          template <typename RandomAccessIterator, typename Real>
          inline set_loop2_inside<RandomAccessIterator, Real> make_set_loop2_inside(
            RandomAccessIterator const one_first, Real const multiplier)
          {
            typedef
              ::ket::mpi::gate::page::set_detail::set_loop2_inside<RandomAccessIterator, Real>
              result_type;

            return result_type(one_first, multiplier);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& set(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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

          typedef
            typename ::ket::utility::meta::real_of<Complex>::type real_type;
          real_type one_probability = static_cast<real_type>(0);

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
            typedef typename boost::range_iterator<page_range_type>::type page_iterator;
            page_iterator const zero_first = boost::begin(zero_page_range);
            page_iterator const one_first = boost::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [&one_probability, zero_first, one_first](StateInteger const index, int const)
              {
                *(zero_first+index) = static_cast<Complex>(0);

                using std::norm;
                one_probability += norm(*(one_first+index));
              });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              ::ket::mpi::gate::page::set_detail::make_set_loop1_inside(
                one_probability, boost::begin(zero_page_range), boost::begin(one_page_range)));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          using std::pow;
          using boost::math::constants::half;
          real_type const multiplier = pow(one_probability, -half<real_type>());

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
              [one_first, multiplier](StateInteger const index, int const)
              { *(one_first+index) *= multiplier; });
# else // BOOST_NO_CXX11_LAMBDAS
            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(one_page_range),
              ::ket::mpi::gate::page::set_detail::make_set_loop2_inside(
                boost::begin(one_page_range), multiplier));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }
      }
    }
  }
}


#endif

