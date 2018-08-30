#ifndef KET_GATE_X_ROTATION_HALF_PI_HPP
# define KET_GATE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <cassert>
# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   include <vector>
# endif
# include <iterator>
# include <utility>
# ifndef NDEBUG
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/is_unsigned.hpp>
#   endif
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

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
# else
#   define KET_is_unsigned boost::is_unsigned
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
  namespace gate
  {
    namespace x_rotation_half_pi_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct x_rotation_half_pi_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        x_rotation_half_pi_loop_inside(
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;

          typedef
            typename std::iterator_traits<RandomAccessIterator>::value_type
            complex_type;
          typedef
            typename ::ket::utility::meta::real_of<complex_type>::type real_type;

          complex_type const zero_iter_value = *zero_iter;

          using boost::math::constants::one_div_root_two;
          *zero_iter += ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter += ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      };

      template <typename RandomAccessIterator, typename StateInteger>
      inline
      x_rotation_half_pi_loop_inside<RandomAccessIterator, StateInteger>
      make_x_rotation_half_pi_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return x_rotation_half_pi_loop_inside<
          RandomAccessIterator, StateInteger>(
            first, qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      void x_rotation_half_pi_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
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

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            typedef
              typename std::iterator_traits<RandomAccessIterator>::value_type
              complex_type;
            complex_type const zero_iter_value = *zero_iter;

            typedef
              typename ::ket::utility::meta::real_of<complex_type>::type real_type;
            using boost::math::constants::one_div_root_two;
            *zero_iter += ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
            *zero_iter *= one_div_root_two<real_type>();
            *one_iter += ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
            *one_iter *= one_div_root_two<real_type>();
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::x_rotation_half_pi_detail::make_x_rotation_half_pi_loop_inside(
            first, qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
      }
    } // namespace x_rotation_half_pi_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void x_rotation_half_pi(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
        ::ket::utility::policy::make_sequential(), first, last, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void x_rotation_half_pi(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
        parallel_policy, first, last, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
          ::ket::utility::policy::make_sequential(),
          boost::begin(state), boost::end(state), qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
          parallel_policy, boost::begin(state), boost::end(state), qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& x_rotation_half_pi(
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit);
        return state;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::x_rotation_half_pi_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges


    namespace x_rotation_half_pi_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct adj_x_rotation_half_pi_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        adj_x_rotation_half_pi_loop_inside(
          RandomAccessIterator const first,
          StateInteger const qubit_mask,
          StateInteger const lower_bits_mask,
          StateInteger const upper_bits_mask)
          : first_(first),
            qubit_mask_(qubit_mask),
            lower_bits_mask_(lower_bits_mask),
            upper_bits_mask_(upper_bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubit, int const) const
        {
          // xxxxx0xxxxxx
          StateInteger const zero_index
            = ((value_wo_qubit bitand upper_bits_mask_) << 1u)
              bitor (value_wo_qubit bitand lower_bits_mask_);
          // xxxxx1xxxxxx
          StateInteger const one_index = zero_index bitor qubit_mask_;
          RandomAccessIterator const zero_iter = first_+zero_index;
          RandomAccessIterator const one_iter = first_+one_index;

          typedef
            typename std::iterator_traits<RandomAccessIterator>::value_type
            complex_type;
          typedef
            typename ::ket::utility::meta::real_of<complex_type>::type real_type;

          complex_type const zero_iter_value = *zero_iter;

          using boost::math::constants::one_div_root_two;
          *zero_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter -= ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        }
      };

      template <typename RandomAccessIterator, typename StateInteger>
      inline
      adj_x_rotation_half_pi_loop_inside<RandomAccessIterator, StateInteger>
      make_adj_x_rotation_half_pi_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return adj_x_rotation_half_pi_loop_inside<
          RandomAccessIterator, StateInteger>(
            first, qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      void adj_x_rotation_half_pi_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        static_assert(
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
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

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          [first, qubit_mask, lower_bits_mask, upper_bits_mask](
            StateInteger const value_wo_qubit, int const)
          {
            // xxxxx0xxxxxx
            StateInteger const zero_index
              = ((value_wo_qubit bitand upper_bits_mask) << 1u)
                bitor (value_wo_qubit bitand lower_bits_mask);
            // xxxxx1xxxxxx
            StateInteger const one_index = zero_index bitor qubit_mask;
            RandomAccessIterator const zero_iter = first+zero_index;
            RandomAccessIterator const one_iter = first+one_index;
            typedef
              typename std::iterator_traits<RandomAccessIterator>::value_type
              complex_type;
            complex_type const zero_iter_value = *zero_iter;

            typedef
              typename ::ket::utility::meta::real_of<complex_type>::type real_type;
            using boost::math::constants::one_div_root_two;
            *zero_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
            *zero_iter *= one_div_root_two<real_type>();
            *one_iter -= ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
            *one_iter *= one_div_root_two<real_type>();
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::x_rotation_half_pi_detail::make_adj_x_rotation_half_pi_loop_inside(
            first, qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
      }
    } // namespace x_rotation_half_pi_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_x_rotation_half_pi(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
        ::ket::utility::policy::make_sequential(), first, last, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_x_rotation_half_pi(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
        parallel_policy, first, last, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
          ::ket::utility::policy::make_sequential(),
          boost::begin(state), boost::end(state), qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
          parallel_policy, boost::begin(state), boost::end(state), qubit);
        return state;
      }

# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
      template <
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_x_rotation_half_pi(
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
          ::ket::utility::policy::make_sequential(),
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit);
        return state;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator,
        typename StateInteger, typename BitInteger>
      inline std::vector<Complex, Allocator>& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        std::vector<Complex, Allocator>& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi_impl(
          parallel_policy,
          KET_addressof(state.front()), KET_addressof(state.front()) + state.size(),
          qubit);
        return state;
      }
# endif // KET_PREFER_POINTER_TO_VECTOR_ITERATOR
    } // namespace ranges
  } // namespace gate
} // namespace ket


# ifdef KET_PREFER_POINTER_TO_VECTOR_ITERATOR
#   undef KET_addressof
# endif
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

