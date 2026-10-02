#ifndef KET_GATE_RESET_HPP
# define KET_GATE_RESET_HPP

# include <iterator>

# include <ket/qubit.hpp>
# include <ket/gate/pauli_x.hpp>
# include <ket/gate/projective_measurement.hpp>


namespace ket
{
  namespace gate
  {
    // RESET_i
    // RESET_i |psi> measures qubit i and maps either outcome to |0>_i.
    template <
      typename ParallelPolicy, typename RandomAccessIterator,
      typename RandomNumberGenerator, typename StateInteger, typename BitInteger>
    inline auto reset(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      RandomNumberGenerator& random_number_generator,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      auto const outcome
        = ::ket::gate::projective_measurement(
            parallel_policy, first, last, random_number_generator, qubit);
      if (outcome == ::ket::gate::outcome::one)
        ::ket::gate::pauli_x(parallel_policy, first, last, qubit);
    }

    template <
      typename RandomAccessIterator, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger>
    inline auto reset(
      RandomAccessIterator const first, RandomAccessIterator const last,
      RandomNumberGenerator& random_number_generator,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    -> void
    {
      ::ket::gate::reset(
        ::ket::utility::policy::make_sequential(), first, last,
        random_number_generator, qubit);
    }

    namespace ranges
    {
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename RandomNumberGenerator, typename StateInteger, typename BitInteger>
      inline auto reset(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::reset(
          parallel_policy, begin(state), end(state),
          random_number_generator, qubit);
        return state;
      }

      template <
        typename RandomAccessRange, typename RandomNumberGenerator,
        typename StateInteger, typename BitInteger>
      inline auto reset(
        RandomAccessRange& state, RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::gate::ranges::reset(
          ::ket::utility::policy::make_sequential(), state,
          random_number_generator, qubit);
      }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_RESET_HPP
