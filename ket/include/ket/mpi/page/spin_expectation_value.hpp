#ifndef KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_PAGE_SPIN_EXPECTATION_VALUE_HPP

# include <cassert>
# include <vector>
# include <array>
# include <iterator>
# include <numeric>
# include <utility>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>
# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator>
      inline
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<RandomAccessRange>::type>::type, 3u>
      spin_expectation_value(
        MpiPolicy const, ParallelPolicy const,
        RandomAccessRange&,
        ::ket::qubit<StateInteger, BitInteger> const,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, Allocator>&)
      {
        using complex_type = typename boost::range_value<RandomAccessRange>::type;
        using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
        static constexpr auto zero_value = real_type{0};

        using spin_type = std::array<real_type, 3u>;
        constexpr auto result = spin_type{zero_value, zero_value, zero_value};

        return result;
      }

      template <
        typename ParallelPolicy,
        typename Complex, typename StateAllocator,
        typename StateInteger, typename BitInteger, typename PermutationAllocator>
      inline
      std::array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
        ::ket::mpi::state<Complex, 0, StateAllocator>&,
        ::ket::qubit<StateInteger, BitInteger> const,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator>&)
      {
        using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
        static constexpr auto zero_value = real_type{0};

        using spin_type = std::array<real_type, 3u>;
        constexpr auto result = spin_type{zero_value, zero_value, zero_value};

        return result;
      }

      template <
        typename ParallelPolicy,
        typename Complex, int num_page_qubits_, typename StateAllocator,
        typename StateInteger, typename BitInteger, typename PermutationAllocator>
      inline
      std::array<typename ::ket::utility::meta::real_of<Complex>::type, 3u>
      spin_expectation_value(
        ::ket::mpi::utility::policy::general_mpi const,
        ParallelPolicy const parallel_policy,
        ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<
          StateInteger, BitInteger, PermutationAllocator>& permutation)
      {
        assert(local_state.is_page_qubit(permutation[qubit]));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto const num_nonpage_qubits
          = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);
        auto const qubit_mask
          = ::ket::utility::integer_exp2<StateInteger>(
              permutation[qubit] - static_cast<qubit_type>(num_nonpage_qubits));
        auto const lower_bits_mask = qubit_mask - StateInteger{1u};
        auto const upper_bits_mask = compl lower_bits_mask;

        using hd_spin_type = std::array<long double, 3u>;
        constexpr auto zero_spin = hd_spin_type{ };
        auto spins_in_threads
          = std::vector<hd_spin_type>(::ket::utility::num_threads(parallel_policy), zero_spin);

        static constexpr auto num_pages
          = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
        for (auto base_page_id = std::size_t{0u}; base_page_id < num_pages/2u; ++base_page_id)
        {
          // x0x
          auto const zero_page_id
            = ((base_page_id bitand upper_bits_mask) << 1u)
              bitor (base_page_id bitand lower_bits_mask);
          // x1x
          auto const one_page_id = zero_page_id bitor qubit_mask;

          auto zero_page_range = local_state.page_range(zero_page_id);
          auto one_page_range = local_state.page_range(one_page_id);
          assert(boost::size(zero_page_range) == boost::size(one_page_range));

          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy,
            boost::size(zero_page_range),
            [&zero_page_range, &one_page_range, &spins_in_threads](
              StateInteger const index, int const thread_index)
            {
              using std::conj;
              auto const conj_zero_value = conj(*(::ket::utility::begin(zero_page_range) + index));
              auto const one_value = *(::ket::utility::begin(one_page_range) + index);
              auto const conj_zero_times_one = conj_zero_value * one_value;

              using std::real;
              spins_in_threads[thread_index][0u] += static_cast<long double>(real(conj_zero_times_one));
              using std::imag;
              spins_in_threads[thread_index][1u] += static_cast<long double>(imag(conj_zero_times_one));
              using std::norm;
              spins_in_threads[thread_index][2u]
                += static_cast<long double>(norm(conj_zero_value)) - static_cast<long double>(norm(one_value));
            });
        }

        auto hd_spin
          = std::accumulate(
              ::ket::utility::begin(spins_in_threads), ::ket::utility::end(spins_in_threads), zero_spin,
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
