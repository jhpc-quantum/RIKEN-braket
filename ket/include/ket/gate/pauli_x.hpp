#ifndef KET_GATE_PAULI_X_HPP
# define KET_GATE_PAULI_X_HPP

# include <boost/config.hpp>

# include <cassert>
# include <algorithm>
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

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
# else
#   define KET_is_unsigned boost::is_unsigned
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace gate
  {
    namespace pauli_x_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct pauli_x_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger qubit_mask_;
        StateInteger lower_bits_mask_;
        StateInteger upper_bits_mask_;

        pauli_x_loop_inside(
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

          std::iter_swap(first_+zero_index, first_+one_index);
        }
      };

      template <typename RandomAccessIterator, typename StateInteger>
      inline
      pauli_x_loop_inside<RandomAccessIterator, StateInteger>
      make_pauli_x_loop_inside(
        RandomAccessIterator const first,
        StateInteger const qubit_mask,
        StateInteger const lower_bits_mask,
        StateInteger const upper_bits_mask)
      {
        return pauli_x_loop_inside<
          RandomAccessIterator, StateInteger>(
            first, qubit_mask, lower_bits_mask, upper_bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      void pauli_x_impl(
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

            std::iter_swap(first+zero_index, first+one_index);
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/2u,
          ::ket::gate::pauli_x_detail::make_pauli_x_loop_inside(
            first, qubit_mask, lower_bits_mask, upper_bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
      }
    } // namespace pauli_x_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::pauli_x_detail::pauli_x_impl(
        ::ket::utility::policy::make_sequential(), first, last, qubit);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::pauli_x_detail::pauli_x_impl(
        parallel_policy, first, last, qubit);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_x_detail::pauli_x_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), qubit);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::pauli_x_detail::pauli_x_impl(
          parallel_policy, ::ket::utility::begin(state), ::ket::utility::end(state), qubit);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_pauli_x(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_x(first, last, qubit); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_pauli_x(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::pauli_x(parallel_policy, first, last, qubit); }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_x(state, qubit); }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::pauli_x(parallel_policy, state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

