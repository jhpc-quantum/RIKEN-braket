#ifndef KET_GATE_X_ROTATION_HALF_PI_HPP
# define KET_GATE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <array>
# include <iterator>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>


namespace ket
{
  namespace gate
  {
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void x_rotation_half_pi(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 2u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0b0u];
          auto const one_iter = first + indices[0b1u];
# else // BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0u];
          auto const one_iter = first + indices[1u];
# endif // BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
          using boost::math::constants::one_div_root_two;
          *zero_iter += ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter += ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        },
        qubit);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void x_rotation_half_pi(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::x_rotation_half_pi(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::x_rotation_half_pi(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::x_rotation_half_pi(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_x_rotation_half_pi(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::gate::gate(
        parallel_policy, first, last,
        [](RandomAccessIterator const first, std::array<StateInteger, 2u> const& indices, int const)
        {
# ifndef BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0b0u];
          auto const one_iter = first + indices[0b1u];
# else // BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter = first + indices[0u];
          auto const one_iter = first + indices[1u];
# endif // BOOST_NO_CXX14_BINARY_LITERALS
          auto const zero_iter_value = *zero_iter;

          using complex_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
          using boost::math::constants::one_div_root_two;
          *zero_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*one_iter);
          *zero_iter *= one_div_root_two<real_type>();
          *one_iter -= ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
          *one_iter *= one_div_root_two<real_type>();
        },
        qubit);
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline void adj_x_rotation_half_pi(
      RandomAccessIterator const first, RandomAccessIterator const last, ::ket::qubit<StateInteger, BitInteger> const qubit)
    { ::ket::gate::adj_x_rotation_half_pi(::ket::utility::policy::make_sequential(), first, last, qubit); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy, RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::gate::adj_x_rotation_half_pi(parallel_policy, std::begin(state), std::end(state), qubit);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& state, ::ket::qubit<StateInteger, BitInteger> const qubit)
      { return ::ket::gate::ranges::adj_x_rotation_half_pi(::ket::utility::policy::make_sequential(), state, qubit); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_X_ROTATION_HALF_PI_HPP
