#ifndef KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP

# include <boost/config.hpp>

# include <cassert>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <iterator>
# include <utility>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/numeric.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator>
      inline
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange>::type>::type, 3u>
      spin_expectation_value(
        MpiPolicy const, ParallelPolicy const,
        RandomAccessRange&,
        ::ket::qubit<StateInteger, BitInteger> const,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, Allocator>&)
      {
        typedef typename boost::range_value<RandomAccessRange>::type complex_type;
        typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        typedef KET_array<real_type, 3u> spin_type;
        real_type BOOST_CONSTEXPR_OR_CONST zero_value = static_cast<real_type>(0);
        spin_type BOOST_CONSTEXPR_OR_CONST result = {zero_value, zero_value, zero_value};
        return result;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename StateAllocator,
        typename StateInteger, typename BitInteger, typename PermutationAllocator>
      inline
      KET_array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
        ::ket::mpi::state<Complex, 0, StateAllocator>&,
        ::ket::qubit<StateInteger, BitInteger> const,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator>&)
      {
        typedef typename ::ket::utility::meta::real_of<Complex>::type real_type;
        typedef KET_array<real_type, 3u> spin_type;
        real_type BOOST_CONSTEXPR_OR_CONST zero_value = static_cast<real_type>(0);
        spin_type BOOST_CONSTEXPR_OR_CONST result = {zero_value, zero_value, zero_value};
        return result;
      }


      namespace spin_expectation_value_detail
      {
# ifdef BOOST_NO_CXX11_LAMBDAS
        template <typename RandomAccessIterator>
        struct spin_expectation_value_loop_inside
        {
          RandomAccessIterator zero_first_;
          RandomAccessIterator one_first_;

          typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
          typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
          typedef KET_array<real_type, 3u> spin_type;
          std::vector<spin_type>& spins_in_threads_;
          std::vector<spin_type>& residuals_in_threads_;

          spin_expectation_value_loop_inside(
            RandomAccessIterator const zero_first,
            RandomAccessIterator const one_first,
            std::vector<spin_type>& spins_in_threads,
            std::vector<spin_type>& residuals_in_threads)
            : zero_first_(zero_first),
              one_first_(one_first),
              spins_in_threads_(spins_in_threads),
              residuals_in_threads_(residuals_in_threads)
          { }

          template <typename StateInteger>
          void operator()(StateInteger const index, int const thread_index) const
          {
            complex_type const zero_value = *(zero_first_+index);
            complex_type const one_value = *(one_first_+index);
            complex_type const zero_times_one = zero_value*one_value;

            using std::real;
            real_type const real_zero_times_one
              = real(zero_times_one) + residuals_in_threads_[thread_index][0u];
            using std::imag;
            real_type const imag_zero_times_one
              = imag(zero_times_one) + residuals_in_threads_[thread_index][1u];
            using std::norm;
            real_type const norm_difference
              = (norm(zero_value) - norm(one_value)) + residuals_in_threads_[thread_index][2u];

            real_type const tmp_x = spins_in_threads_[thread_index][0u] + real_zero_times_one;
            real_type const tmp_y = spins_in_threads_[thread_index][1u] + imag_zero_times_one;
            real_type const tmp_z = spins_in_threads_[thread_index][2u] + norm_difference;

            residuals_in_threads_[thread_index][0u]
              = real_zero_times_one - (tmp_x - spins_in_threads_[thread_index][0u]);
            residuals_in_threads_[thread_index][1u]
              = imag_zero_times_one - (tmp_y - spins_in_threads_[thread_index][1u]);
            residuals_in_threads_[thread_index][2u]
              = norm_difference - (tmp_z - spins_in_threads_[thread_index][2u]);

            spins_in_threads_[thread_index][0u] = tmp_x;
            spins_in_threads_[thread_index][1u] = tmp_y;
            spins_in_threads_[thread_index][2u] = tmp_z;

            /*
            using std::real;
            spins_in_threads_[thread_index][0u] += real(zero_times_one);
            using std::imag;
            spins_in_threads_[thread_index][1u] += imag(zero_times_one);
            using std::norm;
            spins_in_threads_[thread_index][2u] += norm(zero_value) - norm(one_value);
            */
          }
        };

        template <typename RandomAccessIterator, typename Spin>
        inline spin_expectation_value_loop_inside<RandomAccessIterator>
        make_spin_expectation_value_loop_inside(
          RandomAccessIterator const zero_first,
          RandomAccessIterator const one_first,
          std::vector<Spin>& spins_in_threads,
          std::vector<Spin>& residuals_in_threads)
        {
          typedef
            ::ket::mpi::page::spin_expectation_value_detail
              ::spin_expectation_value_loop_inside<RandomAccessIterator>
            result_type;

          return result_type(zero_first, one_first, spins_in_threads, residuals_in_threads);
        }


        template <typename Spin>
        struct spin_expectation_value_accumulate_inside
        {
          Spin& residual_;

          explicit spin_expectation_value_accumulate_inside(Spin& residual)
            : residual_(residual)
          { }

          Spin operator()(Spin const& accumulated_spin, Spin spin) const
          {
            spin[0u] += residual_[0u];
            spin[1u] += residual_[1u];
            spin[2u] += residual_[2u];

            Spin result = accumulated_spin;
            result[0u] += spin[0u];
            result[1u] += spin[1u];
            result[2u] += spin[2u];

            residual_[0u] = spin[0u] - (result[0u] - accumulated_spin[0u]);
            residual_[1u] = spin[1u] - (result[1u] - accumulated_spin[1u]);
            residual_[2u] = spin[2u] - (result[2u] - accumulated_spin[2u]);

            return result;

            /*
            accumulated_spin[0u] += spin_in_thread[0u];
            accumulated_spin[1u] += spin_in_thread[1u];
            accumulated_spin[2u] += spin_in_thread[2u];
            return accumulated_spin;
            */
          }
        };
# endif // BOOST_NO_CXX11_LAMBDAS
      } // namespace spin_expectation_value_detail

      template <
        typename ParallelPolicy,
        typename Complex, int num_page_qubits_, typename StateAllocator,
        typename StateInteger, typename BitInteger, typename PermutationAllocator>
      inline
      KET_array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ::ket::mpi::utility::policy::general_mpi const,
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator>& permutation)
      {
        assert(local_state.is_page_qubit(permutation[qubit]));

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        BitInteger const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
        StateInteger const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(
              permutation[qubit] - static_cast<qubit_type>(num_nonpage_qubits));
        StateInteger const lower_bits_mask = qubit_mask-static_cast<StateInteger>(1u);
        StateInteger const upper_bits_mask = compl lower_bits_mask;

        typedef typename ::ket::utility::meta::real_of<Complex>::type real_type;
        typedef KET_array<real_type, 3u> spin_type;
        spin_type BOOST_CONSTEXPR_OR_CONST zero_spin = { };
        std::vector<spin_type> spins_in_threads(
          ::ket::utility::num_threads(parallel_policy), zero_spin);
        std::vector<spin_type> residuals_in_threads(spins_in_threads.size(), zero_spin);

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

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy,
            boost::size(zero_page_range),
            [&zero_page_range, &one_page_range, &spins_in_threads, &residuals_in_threads](StateInteger const index, int const thread_index)
            {
              Complex const zero_value = *(boost::begin(zero_page_range)+index);
              Complex const one_value = *(boost::begin(one_page_range)+index);
              Complex const zero_times_one = zero_value*one_value;

              using std::real;
              real_type const real_zero_times_one
                = real(zero_times_one) + residuals_in_threads[thread_index][0u];
              using std::imag;
              real_type const imag_zero_times_one
                = imag(zero_times_one) + residuals_in_threads[thread_index][1u];
              using std::norm;
              real_type const norm_difference
                = (norm(zero_value) - norm(one_value)) + residuals_in_threads[thread_index][2u];

              real_type const tmp_x = spins_in_threads[thread_index][0u] + real_zero_times_one;
              real_type const tmp_y = spins_in_threads[thread_index][1u] + imag_zero_times_one;
              real_type const tmp_z = spins_in_threads[thread_index][2u] + norm_difference;

              residuals_in_threads[thread_index][0u]
                = real_zero_times_one - (tmp_x - spins_in_threads[thread_index][0u]);
              residuals_in_threads[thread_index][1u]
                = imag_zero_times_one - (tmp_y - spins_in_threads[thread_index][1u]);
              residuals_in_threads[thread_index][2u]
                = norm_difference - (tmp_z - spins_in_threads[thread_index][2u]);

              spins_in_threads[thread_index][0u] = tmp_x;
              spins_in_threads[thread_index][1u] = tmp_y;
              spins_in_threads[thread_index][2u] = tmp_z;

              /*
              using std::real;
              spins_in_threads[thread_index][0u] += real(zero_times_one);
              using std::imag;
              spins_in_threads[thread_index][1u] += imag(zero_times_one);
              using std::norm;
              spins_in_threads[thread_index][2u] += norm(zero_value) - norm(one_value);
              */
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy,
            boost::size(zero_page_range),
            ::ket::mpi::page::spin_expectation_value_detail::make_spin_expectation_value_loop_inside(
              boost::begin(zero_page_range), boost::begin(one_page_range), spins_in_threads, residuals_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        spin_type residual = zero_spin;
# ifndef BOOST_NO_CXX11_LAMBDAS
        spin_type spin
          = boost::accumulate(
              spins_in_threads, zero_spin,
              [&residual](spin_type const& accumulated_spin, spin_type spin)
              {
                spin[0u] += residual[0u];
                spin[1u] += residual[1u];
                spin[2u] += residual[2u];

                spin_type result = accumulated_spin;
                result[0u] += spin[0u];
                result[1u] += spin[1u];
                result[2u] += spin[2u];

                residual[0u] = spin[0u] - (result[0u] - accumulated_spin[0u]);
                residual[1u] = spin[1u] - (result[1u] - accumulated_spin[1u]);
                residual[2u] = spin[2u] - (result[2u] - accumulated_spin[2u]);

                return result;

                /*
                accumulated_spin[0u] += spin_in_thread[0u];
                accumulated_spin[1u] += spin_in_thread[1u];
                accumulated_spin[2u] += spin_in_thread[2u];
                return accumulated_spin;
                */
              });
# else // BOOST_NO_CXX11_LAMBDAS
        spin_type spin
          = boost::accumulate(
              spins_in_threads, zero_spin,
              ::ket::mpi::page::spin_expectation_value_detail
                ::make_spin_expectation_value_accumulate_inside(residual));
# endif // BOOST_NO_CXX11_LAMBDAS

        using boost::math::constants::half;
        spin[2u] *= half<real_type>();

        return spin;
      }
    } // namespace page
  } // namespace mpi
} // namespace ket


# undef KET_array

#endif
