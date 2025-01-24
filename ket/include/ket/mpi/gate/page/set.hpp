#ifndef KET_MPI_GATE_PAGE_SET_HPP
# define KET_MPI_GATE_PAGE_SET_HPP

# include <cassert>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
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
        // one_probability
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto one_probability(
          ParallelPolicy const,
          RandomAccessRange const& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"one_probability"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto one_probability(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator> const& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> ::ket::utility::meta::real_t<Complex>
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"one_probability"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        inline auto one_probability(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> ::ket::utility::meta::real_t<Complex>
        {
          auto one_probability = 0.0l;

          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&one_probability](auto const, auto const one_first, StateInteger const index, int const)
            {
              using std::norm;
              one_probability += static_cast<long double>(norm(*(one_first + index)));
            });

          using real_type = ::ket::utility::meta::real_t<Complex>;
          return static_cast<real_type>(one_probability);
        }

        // do_set
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto do_set(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          Real const, ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"do_set"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename Real, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto do_set(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          Real const, ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> void
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"do_set"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename Real, typename StateInteger, typename BitInteger>
        inline auto do_set(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          Real const multiplier, ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> void
        {
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


#endif // KET_MPI_GATE_PAGE_SET_HPP
