#ifndef KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP

# include <vector>
# include <array>
# include <numeric>
# include <iterator>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger>
      [[noreturn]] inline
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<RandomAccessRange> >, 3u >
      spin_expectation_value(
        ParallelPolicy const,
        RandomAccessRange&,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
      { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"spin_expectation_value"}; }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
      [[noreturn]] inline
      std::array< ::ket::utility::meta::real_t<Complex>, 3u >
      spin_expectation_value(
        ParallelPolicy const,
        ::ket::mpi::state<Complex, false, Allocator>&,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
      { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"spin_expectation_value"}; }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
      inline
      std::array< ::ket::utility::meta::real_t<Complex>, 3u >
      spin_expectation_value(
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, true, Allocator>& local_state,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
      {
        using hd_spin_type = std::array<long double, 3u>;
        constexpr hd_spin_type zero_spin{ };
        auto spins_in_threads
          = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

        ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
          parallel_policy, local_state, permutated_qubit,
          [&spins_in_threads](auto const zero_first, auto const one_first, StateInteger const index, int const thread_index)
          {
            using std::conj;
            auto const conj_zero_value = conj(*(zero_first + index));
            auto const one_value = *(one_first + index);
            auto const conj_zero_times_one = conj_zero_value * one_value;

            using std::real;
            spins_in_threads[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
            using std::imag;
            spins_in_threads[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
            using std::norm;
            spins_in_threads[thread_index][2u]
              += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
          });

        using std::begin;
        using std::end;
        auto const hd_spin
          = std::accumulate(
              begin(spins_in_threads), end(spins_in_threads), zero_spin,
              [](hd_spin_type accumulated_spin, hd_spin_type const& spin)
              {
                accumulated_spin[0u] += spin[0u];
                accumulated_spin[1u] += spin[1u];
                accumulated_spin[2u] += spin[2u];
                return accumulated_spin;
              });

        using real_type = ::ket::utility::meta::real_t<Complex>;
        using spin_type = std::array<real_type, 3u>;
        spin_type spin{static_cast<real_type>(hd_spin[0u]), static_cast<real_type>(hd_spin[1u]), static_cast<real_type>(hd_spin[2u])};

        using boost::math::constants::half;
        spin[2u] *= half<real_type>();

        return spin;
      }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
