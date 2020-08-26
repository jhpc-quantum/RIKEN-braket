#ifndef KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
# define KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP

# include <cassert>
# include <iterator>

# include <boost/math/constants/constants.hpp>
# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/imaginary_unit.hpp>
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
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& x_rotation_half_pi(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& x_rotation_half_pi(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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
              [zero_first, one_first](StateInteger const index, int const)
              {
                auto const zero_iter = zero_first + index;
                auto const one_iter = one_first + index;
                auto const zero_iter_value = *zero_iter;

                using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
                using boost::math::constants::one_div_root_two;
                *zero_iter += ::ket::utility::imaginary_unit<Complex>() * *one_iter;
                *zero_iter *= one_div_root_two<real_type>();
                *one_iter += ::ket::utility::imaginary_unit<Complex>() * zero_iter_value;
                *one_iter *= one_div_root_two<real_type>();
              });
          }

          return local_state;
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
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
              [zero_first, one_first](StateInteger const index, int const)
              {
                auto const zero_iter = zero_first + index;
                auto const one_iter = one_first + index;
                auto const zero_iter_value = *zero_iter;

                using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
                using boost::math::constants::one_div_root_two;
                *zero_iter -= ::ket::utility::imaginary_unit<Complex>() * *one_iter;
                *zero_iter *= one_div_root_two<real_type>();
                *one_iter -= ::ket::utility::imaginary_unit<Complex>() * zero_iter_value;
                *one_iter *= one_div_root_two<real_type>();
              });
          }

          return local_state;
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
