#ifndef KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP

# include <vector>
# include <array>
# include <numeric>
# include <iterator>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
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
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange>::type>::type, 3u>
      spin_expectation_value(
        ParallelPolicy const,
        RandomAccessRange&,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
      { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"spin_expectation_value"}; }

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
      [[noreturn]] inline
      std::array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ParallelPolicy const,
        ::ket::mpi::state<Complex, false, Allocator>&,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
      { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"spin_expectation_value"}; }

      namespace spin_expectation_value_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename HdSpin>
        struct spin_expectation_value
        {
          std::vector<HdSpin>& spins_in_threads_;

          explicit spin_expectation_value(std::vector<HdSpin>& spins_in_threads)
            : spins_in_threads_{spins_in_threads}
          { }

          template <typename Iterator, typename StateInteger>
          void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const thread_index) const
          {
            using std::conj;
            auto const conj_zero_value = conj(*(zero_first + index));
            auto const one_value = *(one_first + index);
            auto const conj_zero_times_one = conj_zero_value * one_value;

            using std::real;
            spins_in_threads_[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
            using std::imag;
            spins_in_threads_[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
            using std::norm;
            spins_in_threads_[thread_index][2u]
              += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
          }
        }; // struct spin_expectation_value<Complex, Real>

        template <typename HdSpin>
        inline ::ket::mpi::page::spin_expectation_value_detail::spin_expectation_value<HdSpin>
        make_spin_expectation_value(std::vector<HdSpin>& spins_in_threads)
        { return ::ket::mpi::page::spin_expectation_value_detail::spin_expectation_value<HdSpin>{spins_in_threads}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace spin_expectation_value_detail

      template <
        typename ParallelPolicy,
        typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
      inline
      std::array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, true, Allocator>& local_state,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
      {
        using hd_spin_type = std::array<long double, 3u>;
        constexpr auto zero_spin = hd_spin_type{ };
        auto spins_in_threads
          = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
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
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
          parallel_policy, local_state, permutated_qubit,
          ::ket::mpi::page::spin_expectation_value_detail::make_spin_expectation_value<hd_spin_type>(spins_in_threads));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        auto const hd_spin
          = std::accumulate(
              std::begin(spins_in_threads), std::end(spins_in_threads), zero_spin,
              [](hd_spin_type accumulated_spin, hd_spin_type const& spin)
              {
                accumulated_spin[0u] += spin[0u];
                accumulated_spin[1u] += spin[1u];
                accumulated_spin[2u] += spin[2u];
                return accumulated_spin;
              });

        using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
        using spin_type = std::array<real_type, 3u>;
        auto spin = spin_type{};
        spin[0u] = static_cast<real_type>(hd_spin[0u]);
        spin[1u] = static_cast<real_type>(hd_spin[1u]);
        spin[2u] = static_cast<real_type>(hd_spin[2u]);

        using boost::math::constants::half;
        spin[2u] *= half<real_type>();

        return spin;
      }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
