#ifndef KET_MPI_GATE_PAGE_SET_HPP
# define KET_MPI_GATE_PAGE_SET_HPP

# include <cassert>

# include <boost/math/constants/constants.hpp>

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
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto set(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> RandomAccessRange&
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"set"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        [[noreturn]] inline auto set(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        -> ::ket::mpi::state<Complex, false, Allocator>&
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"set"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        inline auto set(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        -> ::ket::mpi::state<Complex, true, Allocator>&
        {
          using real_type = ::ket::utility::meta::real_t<Complex>;
          auto one_probability = real_type{0};

          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&one_probability](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) = Complex{0};

              using std::norm;
              one_probability += norm(*(one_first + index));
            });

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<real_type>());

          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [multiplier](auto const, auto const one_first, StateInteger const index, int const)
            { *(one_first + index) *= multiplier; });
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SET_HPP
