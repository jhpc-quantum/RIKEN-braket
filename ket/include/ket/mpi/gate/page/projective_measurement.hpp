#ifndef KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP

# include <cassert>
# include <cmath>
# include <iterator>
# include <utility>

# include <boost/math/constants/constants.hpp>
# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>


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
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<RandomAccessRange>::type>::type,
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<RandomAccessRange>::type>::type>
        zero_one_probabilities(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<RandomAccessRange>::type>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator> const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<Complex>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);
          auto const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
          auto const lower_bits_mask = qubit_mask - StateInteger{1u};
          auto const upper_bits_mask = compl lower_bits_mask;

          auto zero_probability = 0.0l;
          auto one_probability = 0.0l;

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto base_page_id = std::size_t{0u};
               base_page_id < num_pages / 2u; ++base_page_id)
          {
            // x0x
            auto const zero_page_id
              = ((base_page_id bitand upper_bits_mask) << 1u)
                bitor (base_page_id bitand lower_bits_mask);
            // x1x
            auto const one_page_id = zero_page_id bitor qubit_mask;

            auto zero_page_range = local_state.page_range(zero_page_id);
            auto one_page_range = local_state.page_range(one_page_id);

            auto const zero_first = ::ket::utility::begin(zero_page_range);
            auto const one_first = ::ket::utility::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [&zero_probability, &one_probability, zero_first, one_first](
                StateInteger const index, int const)
              {
                using std::norm;
                zero_probability += static_cast<long double>(norm(*(zero_first + index)));
                one_probability += static_cast<long double>(norm(*(one_first + index)));
              });
          }

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          return std::make_pair(
            static_cast<real_type>(zero_probability), static_cast<real_type>(one_probability));
        }

        // change_state_after_measuring_zero
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_zero(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const zero_probability,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);
          auto const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
          auto const lower_bits_mask = qubit_mask - StateInteger{1u};
          auto const upper_bits_mask = compl lower_bits_mask;

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(zero_probability, -half<Real>());

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto base_page_id = std::size_t{0u};
               base_page_id < num_pages / 2u; ++base_page_id)
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

            auto const zero_first = ::ket::utility::begin(zero_page_range);
            auto const one_first = ::ket::utility::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, multiplier](StateInteger const index, int const)
              {
                *(zero_first + index) *= multiplier;
                *(one_first + index) = Complex{Real{0}};
              });
          }
        }

        // change_state_after_measuring_one
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_one(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const one_probability,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          auto const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits() - num_page_qubits_);
          auto const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<BitInteger>(num_nonpage_qubits));
          auto const lower_bits_mask = qubit_mask - StateInteger{1u};
          auto const upper_bits_mask = compl lower_bits_mask;

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<Real>());

          static constexpr auto num_pages
            = ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>::num_pages;
          for (auto base_page_id = std::size_t{0u};
               base_page_id < num_pages / 2u; ++base_page_id)
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

            auto const zero_first = ::ket::utility::begin(zero_page_range);
            auto const one_first = ::ket::utility::begin(one_page_range);

            using ::ket::utility::loop_n;
            loop_n(
              parallel_policy,
              boost::size(zero_page_range),
              [zero_first, one_first, multiplier](StateInteger const index, int const)
              {
                *(zero_first + index) = Complex{Real{0}};
                *(one_first + index) *= multiplier;
              });
          }
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
