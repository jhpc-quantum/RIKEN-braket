#ifndef KET_GATE_TOFFOLI_HPP
# define KET_GATE_TOFFOLI_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
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

# include <boost/math/constants/constants.hpp>

# include <boost/range/algorithm/sort.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

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
    namespace toffoli_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename RandomAccessIterator, typename StateInteger>
      struct toffoli_loop_inside
      {
        RandomAccessIterator first_;
        StateInteger target_qubit_mask_;
        StateInteger control_qubits_mask_;
        KET_array<StateInteger, 4u> const& bits_mask_;

        toffoli_loop_inside(
          RandomAccessIterator const first,
          StateInteger const target_qubit_mask,
          StateInteger const control_qubits_mask,
          KET_array<StateInteger, 4u> const& bits_mask)
          : first_(first),
            target_qubit_mask_(target_qubit_mask),
            control_qubits_mask_(control_qubits_mask),
            bits_mask_(bits_mask)
        { }

        void operator()(StateInteger const value_wo_qubits, int const) const
        {
          // xxx0_cxxx0_txxxx0_cxxx
          StateInteger const base_index
            = ((value_wo_qubits bitand bits_mask_[3u]) << 3u)
              bitor ((value_wo_qubits bitand bits_mask_[2u]) << 2u)
              bitor ((value_wo_qubits bitand bits_mask_[1u]) << 1u)
              bitor (value_wo_qubits bitand bits_mask_[0u]);
          // xxx1_cxxx0_txxxx1_cxxx
          StateInteger const control_on_index
            = base_index bitor control_qubits_mask_;
          // xxx1_cxxx1_txxxx1_cxxx
          StateInteger const target_control_on_index
            = control_on_index bitor target_qubit_mask_;
    
          std::iter_swap(
            first_+control_on_index, first_+target_control_on_index);
        }
      };

      template <typename RandomAccessIterator, typename StateInteger>
      inline toffoli_loop_inside<
        RandomAccessIterator, StateInteger>
      make_toffoli_loop_inside(
        RandomAccessIterator const first,
        StateInteger const target_qubit_mask,
        StateInteger const control_qubits_mask,
        KET_array<StateInteger, 4u> const& bits_mask)
      {
        return toffoli_loop_inside<RandomAccessIterator, StateInteger>(
          first, target_qubit_mask, control_qubits_mask, bits_mask);
      }
# endif // BOOST_NO_CXX11_LAMBDAS

      template <
        typename ParallelPolicy, typename RandomAccessIterator,
        typename StateInteger, typename BitInteger>
      void toffoli_impl(
        ParallelPolicy const parallel_policy,
        RandomAccessIterator const first, RandomAccessIterator const last,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      {
        static_assert(
          KET_is_unsigned<StateInteger>::value,
          "StateInteger should be unsigned");
        static_assert(
          KET_is_unsigned<BitInteger>::value,
          "BitInteger should be unsigned");
        assert(
          ::ket::utility::integer_exp2<StateInteger>(target_qubit)
            < static_cast<StateInteger>(last-first)
          and ::ket::utility::integer_exp2<StateInteger>(control_qubit1.qubit())
                < static_cast<StateInteger>(last-first)
          and ::ket::utility::integer_exp2<StateInteger>(control_qubit2.qubit())
                < static_cast<StateInteger>(last-first)
          and target_qubit != control_qubit1.qubit()
          and target_qubit != control_qubit2.qubit()
          and control_qubit1.qubit() != control_qubit2.qubit());
        assert(
          ::ket::utility::integer_exp2<StateInteger>(
            ::ket::utility::integer_log2<BitInteger>(last-first))
          == static_cast<StateInteger>(last-first));

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        KET_array<qubit_type, 3u> sorted_qubits
          = {target_qubit, control_qubit1.qubit(), control_qubit2.qubit()};
        boost::sort(sorted_qubits);

        StateInteger const target_qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
        StateInteger const control_qubits_mask
          = ::ket::utility::integer_exp2<StateInteger>(control_qubit1.qubit())
            bitor ::ket::utility::integer_exp2<StateInteger>(control_qubit2.qubit());

        KET_array<StateInteger, 4u> bits_mask;
        bits_mask[0u]
          = ::ket::utility::integer_exp2<StateInteger>(sorted_qubits[0u])
            - static_cast<StateInteger>(1u);
        bits_mask[1u]
          = (::ket::utility::integer_exp2<StateInteger>(
               sorted_qubits[1u]-static_cast<BitInteger>(1u))
             - static_cast<StateInteger>(1u))
            xor bits_mask[0u];
        bits_mask[2u]
          = (::ket::utility::integer_exp2<StateInteger>(
               sorted_qubits[2u]-static_cast<BitInteger>(2u))
             - static_cast<StateInteger>(1u))
            xor (bits_mask[0u] bitor bits_mask[1u]);
        bits_mask[3u]
          = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/8u,
          [first, target_qubit_mask, control_qubits_mask, &bits_mask](
            StateInteger const value_wo_qubits, int const)
          {
            // xxx0_cxxx0_txxxx0_cxxx
            StateInteger const base_index
              = ((value_wo_qubits bitand bits_mask[3u]) << 3u)
                bitor ((value_wo_qubits bitand bits_mask[2u]) << 2u)
                bitor ((value_wo_qubits bitand bits_mask[1u]) << 1u)
                bitor (value_wo_qubits bitand bits_mask[0u]);
            // xxx1_cxxx0_txxxx1_cxxx
            StateInteger const control_on_index = base_index bitor control_qubits_mask;
            // xxx1_cxxx1_txxxx1_cxxx
            StateInteger const target_control_on_index
              = control_on_index bitor target_qubit_mask;
      
            std::iter_swap(
              first+control_on_index, first+target_control_on_index);
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy,
          static_cast<StateInteger>(last-first)/8u,
          ::ket::gate::toffoli_detail::make_toffoli_loop_inside(
            first, target_qubit_mask, control_qubits_mask, bits_mask));
# endif // BOOST_NO_CXX11_LAMBDAS
      }
    } // namespace toffoli_detail

    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    {
      ::ket::gate::toffoli_detail::toffoli_impl(
        ::ket::utility::policy::make_sequential(),
        first, last, target_qubit, control_qubit1, control_qubit2);
    }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    {
      ::ket::gate::toffoli_detail::toffoli_impl(
        parallel_policy, first, last, target_qubit, control_qubit1, control_qubit2);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      {
        ::ket::gate::toffoli_detail::toffoli_impl(
          ::ket::utility::policy::make_sequential(),
          ::ket::utility::begin(state), ::ket::utility::end(state), target_qubit, control_qubit1, control_qubit2);
        return state;
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      {
        ::ket::gate::toffoli_detail::toffoli_impl(
          parallel_policy,
          ::ket::utility::begin(state), ::ket::utility::end(state), target_qubit, control_qubit1, control_qubit2);
        return state;
      }
    } // namespace ranges


    template <
      typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    { ::ket::gate::toffoli(first, last, target_qubit, control_qubit1, control_qubit2); }

    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename StateInteger, typename BitInteger>
    inline void adj_toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    {
      ::ket::gate::toffoli(
        parallel_policy, first, last, target_qubit, control_qubit1, control_qubit2);
    }

    namespace ranges
    {
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      { return ::ket::gate::ranges::toffoli(state, target_qubit, control_qubit1, control_qubit2); }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      {
        return ::ket::gate::ranges::toffoli(
          parallel_policy, state, target_qubit, control_qubit1, control_qubit2);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


# undef KET_array
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

