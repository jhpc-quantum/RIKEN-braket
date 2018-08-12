#ifndef KET_SPIN_EXPECTATION_VALUE_HPP
# define KET_SPIN_EXPECTATION_VALUE_HPP

# include <boost/config.hpp>

# include <cassert>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# include <vector>
# include <iterator>
# include <utility>
# ifndef NDEBUG
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/is_unsigned.hpp>
#     include <boost/utility/enable_if.hpp>
#   endif
# endif
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <boost/math/constants/constants.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/numeric.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_enable_if std::enable_if
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define KET_addressof std::addressof
#   else
#     define KET_addressof boost::addressof
#   endif
# endif


namespace ket
{
  namespace spin_expectation_value_detail
  {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct spin_expectation_value_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
        typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        typedef KET_array<real_type, 3u> spin_type;
        std::vector<spin_type>& spins_in_threads_;
        std::vector<spin_type>& residuals_in_threads_;

        spin_expectation_value_loop_inside(
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask,
          std::vector<spin_type>& spins_in_threads,
          std::vector<spin_type>& residuals_in_threads)
          : first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask),
            spins_in_threads_(spins_in_threads),
            residuals_in_threads_(residuals_in_threads)
        { }

        void operator()(StateInteger const value_wo_qubit, int const thread_index) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;

          complex_type const zero_value = *(first_+zero_index);
          complex_type const one_value = *(first_+one_index);
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

      template <typename RandomAccessIterator, typename StateInteger, typename Spin>
      inline spin_expectation_value_loop_inside<RandomAccessIterator, StateInteger>
      make_spin_expectation_value_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask,
        std::vector<Spin>& spins_in_threads,
        std::vector<Spin>& residuals_in_threads)
      {
        return spin_expectation_value_loop_inside<RandomAccessIterator, StateInteger>(
          first, qubit_mask, lower_bits_mask, upper_bits_mask, spins_in_threads, residuals_in_threads);
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

      template <typename Spin>
      inline spin_expectation_value_accumulate_inside<Spin>
      make_spin_expectation_value_accumulate_inside(Spin& residual)
      { return spin_expectation_value_accumulate_inside<Spin>(residual); }
# endif // BOOST_NO_CXX11_LAMBDAS
  } // namespace spin_expectation_value_detail


  template <
    typename ParallelPolicy,
    typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline
  KET_array<
    typename ::ket::utility::meta::real_of<
      typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u>
  spin_expectation_value(
    ParallelPolicy const parallel_policy,
    RandomAccessIterator const first, RandomAccessIterator const last,
    ::ket::qubit<StateInteger, BitInteger> const qubit)
  {
    static_assert(KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
    static_assert(KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
    assert(
      ::ket::utility::integer_exp2<StateInteger>(qubit)
      < static_cast<StateInteger>(last-first));
    assert(
      ::ket::utility::integer_exp2<StateInteger>(
        ::ket::utility::integer_log2<BitInteger>(last-first))
      == static_cast<StateInteger>(last-first));

    StateInteger const qubit_mask
      = ::ket::utility::integer_exp2<StateInteger>(qubit);
    StateInteger const lower_bits_mask
      = qubit_mask-static_cast<StateInteger>(1u);
    StateInteger const upper_bits_mask = compl lower_bits_mask;

    typedef typename std::iterator_traits<RandomAccessIterator>::value_type complex_type;
    typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
    typedef KET_array<real_type, 3u> spin_type;
    spin_type BOOST_CONSTEXPR_OR_CONST zero_spin = { };
    std::vector<spin_type> spins_in_threads(
      ::ket::utility::num_threads(parallel_policy), zero_spin);
    std::vector<spin_type> residuals_in_threads(spins_in_threads.size(), zero_spin);

    using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
    loop_n(
      parallel_policy,
      static_cast<StateInteger>(last-first)/2u,
      [first, qubit_mask, lower_bits_mask, upper_bits_mask,
       &spins_in_threads, &residuals_in_threads](
        StateInteger const value_wo_qubit, int const thread_index)
      {
        // xxxxx0xxxxxx
        StateInteger const zero_index
          = ((value_wo_qubit bitand upper_bits_mask) << 1u)
            bitor (value_wo_qubit bitand lower_bits_mask);
        // xxxxx1xxxxxx
        StateInteger const one_index = zero_index bitor qubit_mask;

        complex_type const zero_value = *(first+zero_index);
        complex_type const one_value = *(first+one_index);
        complex_type const zero_times_one = zero_value*one_value;

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
      static_cast<StateInteger>(last-first)/2u,
      ::ket::spin_expectation_value_detail::make_spin_expectation_value_loop_inside(
        first, qubit_mask, lower_bits_mask, upper_bits_mask, spins_in_threads, residuals_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS

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
          ::ket::spin_expectation_value_detail::make_spin_expectation_value_accumulate_inside(residual));
# endif // BOOST_NO_CXX11_LAMBDAS

    using boost::math::constants::half;
    spin[2u] *= half<real_type>();

    return spin;
  }

  template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
  inline
  typename KET_enable_if<
    not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessIterator>::value,
    KET_array<
      typename ::ket::utility::meta::real_of<
        typename std::iterator_traits<RandomAccessIterator>::value_type>::type, 3u> >::type
  spin_expectation_value(
    RandomAccessIterator const first, RandomAccessIterator const last,
    ::ket::qubit<StateInteger, BitInteger> const qubit)
  { return ::ket::spin_expectation_value(::ket::utility::policy::make_sequential(), first, last, qubit); }

  namespace ranges
  {
    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange const>::type>::type, 3u> >::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      RandomAccessRange const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { return ::ket::spin_expectation_value(parallel_policy, boost::begin(state), boost::end(state), qubit); }

    template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
    inline
    KET_array<
      typename ::ket::utility::meta::real_of<
        typename boost::range_value<RandomAccessRange const>::type>::type, 3u>
    spin_expectation_value(
      RandomAccessRange const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::spin_expectation_value(
        ::ket::utility::policy::make_sequential(), boost::begin(state), boost::end(state), qubit);
    }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    template <
      typename ParallelPolicy,
      typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
    inline KET_array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      std::vector<Complex, Allocator> const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::spin_expectation_value(
        parallel_policy,
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
        qubit);
    }

    template <
      typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
    inline KET_array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
    spin_expectation_value(
      std::vector<Complex, Allocator> const& state,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::spin_expectation_value(
        ::ket::utility::policy::make_sequential(),
        KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
        qubit);
    }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
  } // namespace ranges
} // namespace ket


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_enable_if
# undef KET_is_unsigned
# undef KET_array
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif
