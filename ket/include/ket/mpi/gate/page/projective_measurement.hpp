#ifndef KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP

# include <cmath>
# include <utility>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // zero_one_probabilities
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto zero_one_probabilities(
          ParallelPolicy const,
          RandomAccessRange const& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> std::pair<
             ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >,
             ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> > >
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"zero_one_probabilities"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto zero_one_probabilities(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator> const& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> std::pair< ::ket::utility::meta::real_t<Complex>, ::ket::utility::meta::real_t<Complex> >
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"zero_one_probabilities"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        inline auto zero_one_probabilities(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> std::pair< ::ket::utility::meta::real_t<Complex>, ::ket::utility::meta::real_t<Complex> >
        {
          auto zero_probability = 0.0l;
          auto one_probability = 0.0l;

          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&zero_probability, &one_probability](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              using std::norm;
              zero_probability += static_cast<long double>(norm(*(zero_first + index)));
              one_probability += static_cast<long double>(norm(*(one_first + index)));
            });

          using real_type = ::ket::utility::meta::real_t<Complex>;
          return std::make_pair(static_cast<real_type>(zero_probability), static_cast<real_type>(one_probability));
        }

        // change_state_after_measuring_zero
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Real>
        [[noreturn]] inline auto change_state_after_measuring_zero(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
          Real const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"change_state_after_measuring_zero"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger, typename Real>
        [[noreturn]] inline auto change_state_after_measuring_zero(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
          Real const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"change_state_after_measuring_zero"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger, typename Real>
        inline auto change_state_after_measuring_zero(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          Real const zero_probability)
        -> void
        {
          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(zero_probability, -half<Real>());

          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [multiplier](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) *= multiplier;
              *(one_first + index) = Complex{Real{0}};
            });
        }

        // change_state_after_measuring_one
        template <
          typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real>
        [[noreturn]] inline auto change_state_after_measuring_one(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
          Real const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"change_state_after_measuring_one"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger, typename Real>
        [[noreturn]] inline auto change_state_after_measuring_one(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
          Real const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"change_state_after_measuring_one"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger, typename Real>
        inline auto change_state_after_measuring_one(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          Real const one_probability)
        -> void
        {
          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<Real>());

          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [multiplier](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) = Complex{Real{0}};
              *(one_first + index) *= multiplier;
            });
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
