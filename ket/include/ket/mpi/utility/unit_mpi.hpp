#ifndef KET_MPI_UTILITY_UNIT_MPI_HPP
# define KET_MPI_UTILITY_UNIT_MPI_HPP

# include <cstddef>
# include <cassert>
# ifndef NDEBUG
#   include <iostream>
# endif
# include <vector>
# include <algorithm>
# include <numeric>
# include <iterator>
# include <utility>
# include <array>
# include <type_traits>

# include <boost/range/value_type.hpp>
# include <boost/range/size.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# ifndef NDEBUG
#   include <yampi/lowest_io_process.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# ifndef NDEBUG
#   include <ket/mpi/page/is_on_page.hpp>
#   include <ket/mpi/page/are_on_nonpage.hpp>
# endif
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/detail/make_local_swap_qubit.hpp>
# include <ket/mpi/utility/detail/swap_permutated_local_qubits.hpp>
# include <ket/mpi/utility/detail/interchange_qubits.hpp>
# include <ket/mpi/utility/detail/for_each_in_diagonal_loop.hpp>
# include <ket/mpi/utility/detail/swap_local_data.hpp>
# include <ket/mpi/utility/logger.hpp>
# ifndef NDEBUG
#   include <ket/qubit_io.hpp>
#   include <ket/mpi/qubit_permutation_io.hpp>
# endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace policy
      {
        /*
         * qubit index: xxxxx|xxxxxx|xxxxxxxxx, global qubits, unit qubits, and local qubits from left to right
         * N = L + K + M: the number of qubits
         * L: the number of local qubits, l: value of local qubits
         * K: the number of unit qubits, u: value of unit qubits
         * M: the number of global qubits, g: value of global qubits
         * Each unit has n_u MPI processes, and the value of global qubits is unit index.
         * The total number of MPI processes is 2^M n_u.
         *
         * Let k be the *expected* number of data blocks, and r_u = u / k be an index of an MPI process in a unit.
         * Note that r_u also satisfies r_u = r % n_u if a rank of an MPI process is r.
         * Actual rank of the MPI process is given by r = g n_u + r_u.
         * Moreover, element index in the process is given by i = i_u * 2^L + l, where i_u = u % k.
         * Note that u = k r_u + i_u.
         * The number of data blocks in each process, k~, is k if 0 <= r_u < n_u-1, and 2^K - (n_u-1) k if r_u = n_u-1.
         * Because the number of elements should be as close as possible, we obtain ideal k, k* = 2^K / n_u.
         * We set an integer k to be closest to k^* for given K.
         */
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        class unit_mpi
        {
          static_assert(
            std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(
            std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(
            std::is_unsigned<NumProcesses>::value, "NumProcesses should be unsigned");

          BitInteger num_unit_qubits_; // K
          NumProcesses num_processes_per_unit_; // n_u

          StateInteger expected_num_data_blocks_; // k

         public:
          unit_mpi(
            BitInteger const num_unit_qubits, NumProcesses const num_processes_per_unit,
            yampi::communicator const& communicator, yampi::environment const& environment)
            : num_unit_qubits_{num_unit_qubits},
              num_processes_per_unit_{num_processes_per_unit},
              expected_num_data_blocks_{
                generate_expected_num_data_blocks(num_unit_qubits, num_processes_per_unit)}
          {
            assert(num_unit_qubits >= BitInteger{1u});
            assert(
              num_processes_per_unit >= NumProcesses{1u}
              and num_processes_per_unit <= ::ket::utility::integer_exp2<NumProcesses>(num_unit_qubits));
          }

          // K
          BitInteger const& num_unit_qubits() const noexcept { return num_unit_qubits_; }
          // n_u
          NumProcesses const& num_processes_per_unit() const noexcept { return num_processes_per_unit_; }

          // k
          StateInteger const& expected_num_data_blocks() const noexcept { return expected_num_data_blocks_; }

         private:
          StateInteger generate_expected_num_data_blocks(
            BitInteger const num_unit_qubits, NumProcesses const num_processes_per_unit) const noexcept
          {
            // k* = 2^K / n_u.
            auto const ideal_num_data_blocks
              = static_cast<double>(::ket::utility::integer_exp2<StateInteger>(num_unit_qubits))
                / static_cast<double>(num_processes_per_unit);

            auto const integral_part = static_cast<StateInteger>(ideal_num_data_blocks);
            auto const fractional_part = ideal_num_data_blocks - static_cast<double>(integral_part);
            auto const result
              = fractional_part < 0.5
                ? integral_part
                : integral_part + StateInteger{1u};

            assert(
              result * (num_processes_per_unit - NumProcesses{1u})
              < ::ket::utility::integer_exp2<NumProcesses>(num_unit_qubits));

            return result;
          }
        }; // class unit_mpi<StateInteger, BitInteger, NumProcesses>

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> make_unit_mpi(
          BitInteger const num_unit_qubits, NumProcesses const num_unit_processes,
          yampi::communicator const& communicator, yampi::environment const& environment) noexcept
        { return { num_unit_qubits, num_unit_processes, communicator, environment }; }

        namespace meta
        {
          template <typename T>
          struct is_mpi_policy;

          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct is_mpi_policy< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
            : std::true_type
          { }; // struct is_mpi_policy< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace meta

        // 2^K
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_unit_qubit_values(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy)
        { return ::ket::utility::integer_exp2<StateInteger>(mpi_policy.num_unit_qubits()); }

        // r_u = r % n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline yampi::rank rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::rank const rank)
        {
          assert(rank.mpi_rank() >= 0);
          return rank % mpi_policy.num_processes_per_unit();
        }

        // r_u = r % n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline yampi::rank rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        { return ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator.rank(environment)); }

        // r_u = u / k
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline yampi::rank rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const unit_qubit_value)
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          auto const result
            = static_cast<int>(unit_qubit_value / mpi_policy.expected_num_data_blocks());
          assert(result < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return yampi::rank{result};
        }

        // i_u = u % k
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger data_block_index(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const unit_qubit_value)
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return unit_qubit_value % mpi_policy.expected_num_data_blocks();
        }

        // k~
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_data_blocks(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::rank const rank_in_unit)
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));

          // k (if 0 <= r_u < n_u-1), 2^K - (n_u-1) k (if r_u = n_u-1)
          return rank_in_unit == yampi::rank{static_cast<int>(mpi_policy.num_processes_per_unit()) - 1}
            ? ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy)
              - (static_cast<StateInteger>(mpi_policy.num_processes_per_unit()) - StateInteger{1u})
                * mpi_policy.expected_num_data_blocks()
            : mpi_policy.expected_num_data_blocks();
        }

        // k~
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_data_blocks(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::policy::num_data_blocks(
            mpi_policy,
            ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
        }

        // u = k r_u + i_u.
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger unit_qubit_value(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const data_block_index, yampi::rank const rank_in_unit)
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          assert(data_block_index < ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));
          auto const result
            = mpi_policy.expected_num_data_blocks() * rank_in_unit.mpi_rank() + data_block_index;
          assert(result < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return result;
        }

        // 2^M
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_units(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const result
            = static_cast<StateInteger>(communicator.size(environment))
              / static_cast<StateInteger>(mpi_policy.num_processes_per_unit());
          assert(
            result * mpi_policy.num_processes_per_unit() == static_cast<StateInteger>(communicator.size(environment))
            and ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(result)) == result);
          return result;
        }

        // M
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_global_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
          StateInteger const num_units)
        {
          assert(num_units >= StateInteger{1u});
          auto const result = ::ket::utility::integer_log2<BitInteger>(num_units);
          assert(::ket::utility::integer_exp2<StateInteger>(result) == num_units);
          return result;
        }

        // M
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger num_global_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::policy::num_global_qubits(
            mpi_policy,
            ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
        }

        // g = r / n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger global_qubit_value(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::rank const rank)
        {
          assert(rank.mpi_rank() >= 0);
          return
            static_cast<StateInteger>(rank.mpi_rank())
            / static_cast<StateInteger>(mpi_policy.num_processes_per_unit());
        }

        // g = r / n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline StateInteger global_qubit_value(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const result
            = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator.rank(environment));
          assert(result < ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
          return result;
        }

        // r = g n_u + r_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline yampi::rank rank(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const global_qubit_value, yampi::rank const rank_in_unit)
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return global_qubit_value * mpi_policy.num_processes_per_unit() + rank_in_unit;
        }

        // r = g n_u + r_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline yampi::rank rank(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const global_qubit_value,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          assert(global_qubit_value < ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
          return ::ket::mpi::utility::policy::rank(
            mpi_policy, global_qubit_value,
            ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
        }

        // 2^L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline StateInteger data_block_size(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, yampi::rank const rank_in_unit)
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          assert(boost::size(local_state) % ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit) == 0u);
          auto const result
            = static_cast<StateInteger>(
                boost::size(local_state) / ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));
          assert(::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(result)) == result);
          return result;
        }

        // 2^L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline StateInteger data_block_size(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::policy::data_block_size(
            mpi_policy, local_state,
            ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
        }

        // 2^L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline StateInteger data_block_size(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, StateInteger const unit_qubit_value)
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return ::ket::mpi::utility::policy::data_block_size(
            mpi_policy, local_state,
            ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value));
        }

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline BitInteger num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
          StateInteger const data_block_size)
        {
          assert(data_block_size >= StateInteger{2u});
          auto const result = ::ket::utility::integer_log2<BitInteger>(data_block_size);
          assert(::ket::utility::integer_exp2<StateInteger>(result) == data_block_size);
          return result;
        }

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline BitInteger num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, yampi::rank const rank_in_unit)
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return ::ket::mpi::utility::policy::num_local_qubits(
            mpi_policy,
            ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
        }

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline BitInteger num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::utility::policy::num_local_qubits(
            mpi_policy,
            ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment));
        }

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline BitInteger num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, StateInteger const unit_qubit_value)
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return ::ket::mpi::utility::policy::num_local_qubits(
            mpi_policy,
            ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, unit_qubit_value));
        }
      } // namespace policy

      namespace dispatch
      {
        template <std::size_t num_qubits_of_operation, typename MpiPolicy>
        struct maybe_interchange_qubits;

        template <
          std::size_t num_qubits_of_operation,
          typename StateInteger, typename BitInteger, typename NumProcesses>
        struct maybe_interchange_qubits<
          num_qubits_of_operation,
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);

            auto const num_local_qubits
              = ::ket::mpi::utility::policy::num_local_qubits(
                  mpi_policy, local_state, communicator, environment);

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto permutated_nonlocal_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_nonlocal_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_nonlocal_swap_qubits[index] >= permutated_qubit_type{num_local_qubits})
                continue;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, communicator, environment);
              return;
            }

            do_call(
              mpi_policy, parallel_policy, local_state, num_local_qubits,
              permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, buffer, communicator, environment,
              [&buffer](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  target_rank, communicator, environment);
              });
          }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);

            auto const num_local_qubits
              = ::ket::mpi::utility::policy::num_local_qubits(
                  mpi_policy, local_state, communicator, environment);

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto permutated_nonlocal_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_nonlocal_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_nonlocal_swap_qubits[index] >= permutated_qubit_type{num_local_qubits})
                continue;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);
              return;
            }

            do_call(
              mpi_policy, parallel_policy, local_state, num_local_qubits,
              permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, buffer, communicator, environment,
              [&buffer, &datatype](
                LocalState& local_state,
                StateInteger const data_block_index, StateInteger const data_block_size,
                StateInteger const source_local_first_index, StateInteger const source_local_last_index,
                yampi::rank const target_rank,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                ::ket::mpi::utility::detail::interchange_qubits(
                  local_state, buffer,
                  data_block_index, data_block_size,
                  source_local_first_index, source_local_last_index,
                  datatype, target_rank, communicator, environment);
              });
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename Function>
          static void do_call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            BitInteger const num_local_qubits,
            std::array<
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >,
              num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            Function&& interchange_qubits)
          {
            ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, '>'), environment};

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            // (ex.: num_qubits_of_operation == 3)
            //  Swaps between xxbxb'x|b''xx|cc'c''xxxxxxxx and
            // xxcxc'x|c''xx|bb'b''xxxxxxxx (c = b or ~b). Upper, middle, and
            // lower qubits and are global, unit, and local qubits,
            // respectively. The first three upper qubits in the local qubits
            // are "local swap qubits". Three bits in global qubits and the
            // "local swap qubits" would be swapped.

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto permutated_local_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              permutated_local_swap_qubits[index]
                = permutated_qubit_type{num_local_qubits - BitInteger{1u} - static_cast<BitInteger>(index)};
              local_swap_qubits[index]
                = ::ket::mpi::utility::detail::make_local_swap_qubit(
                    mpi_policy, parallel_policy, local_state, permutation,
                    unswappable_qubits, permutated_local_swap_qubits[index],
                    ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment),
                    ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment),
                    communicator, environment);
            }

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            auto const least_global_permutated_qubit = permutated_qubit_type{num_local_qubits + mpi_policy.num_unit_qubits()};
            auto const num_permutated_unit_swap_qubits
              = std::count_if(
                  std::begin(permutated_nonlocal_swap_qubits),
                  std::end(permutated_nonlocal_swap_qubits),
                  [least_global_permutated_qubit](permutated_qubit_type const permutated_qubit)
                  { return permutated_qubit < least_global_permutated_qubit; });

            auto const data_block_size
              = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);

            if (num_permutated_unit_swap_qubits == 0)
            {
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
              auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit);

              // xbxb'xb''x(|xxxx|xxxxxxxx)
              auto const source_global_qubit_value
                = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment);

              auto const last_global_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
              for (auto global_qubit_mask = StateInteger{1u};
                   global_qubit_mask < last_global_qubit_mask; ++global_qubit_mask)
              {
                // xcxc'xc''x(|xxxx|xxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
                auto mask = StateInteger{0u};
                for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                  mask
                    |= ((global_qubit_mask bitand (StateInteger{1u} << index)) >> index)
                       << (permutated_nonlocal_swap_qubits[index] - least_global_permutated_qubit);
                auto const target_global_qubit_value = source_global_qubit_value xor mask;
                auto const target_rank
                  = ::ket::mpi::utility::policy::rank(mpi_policy, target_global_qubit_value, rank_in_unit);

                // (0000000|0000|)cc'c''00000
                auto source_local_first_index = StateInteger{0u};
                for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                  source_local_first_index
                    |= ((target_global_qubit_value << least_global_permutated_qubit)
                        bitand (StateInteger{1u} << permutated_nonlocal_swap_qubits[index]))
                       >> (permutated_nonlocal_swap_qubits[index] - permutated_local_swap_qubits[index]);

                // (0000000|0000|)cc'c''11111 + 1
                auto const source_local_last_index
                  = source_local_first_index + (data_block_size >> num_qubits_of_operation);

                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  ::ket::mpi::utility::log_with_time_guard<char> print{
                    ::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, ">::swap"),
                    environment};

                  interchange_qubits(
                    local_state,
                    data_block_index, data_block_size,
                    source_local_first_index, source_local_last_index,
                    target_rank, communicator, environment);
                }
              }
            }
            else // num_permutated_unit_swap_qubits != 0
            {
              auto const present_rank = communicator.rank(environment);

              // initialization of permutated_nonlocal_qubit_index_pairs ({sorted_nonlocal_qubit, corresponding_index_in_some_arrays}, ...)
              using permutated_nonlocal_qubit_index_pair_type = std::pair<permutated_qubit_type, std::size_t>;
              auto permutated_nonlocal_qubit_index_pairs
                = std::array<permutated_nonlocal_qubit_index_pair_type, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                permutated_nonlocal_qubit_index_pairs[index]
                  = std::make_pair(permutated_nonlocal_swap_qubits[index], index);

              std::sort(
                std::begin(permutated_nonlocal_qubit_index_pairs),
                std::end(permutated_nonlocal_qubit_index_pairs),
                [](permutated_nonlocal_qubit_index_pair_type const& lhs, permutated_nonlocal_qubit_index_pair_type const& rhs)
                { return lhs.first < rhs.first;});

              // initialization of permutated_nonlocal_qubit_masks (000001000000, 000000001000, 001000000000)
              auto permutated_nonlocal_qubit_masks = std::array<StateInteger, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                permutated_nonlocal_qubit_masks[index]
                  = (StateInteger{1u} << permutated_nonlocal_swap_qubits[index]) >> num_local_qubits;

              // initialization of nonlocal_qubit_value_masks (000000xxx, 0000xx000, 00xx00000, xx0000000)
              auto nonlocal_qubit_value_masks
                = std::array<StateInteger, num_qubits_of_operation + std::size_t{1u}>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                nonlocal_qubit_value_masks[index]
                  = (permutated_nonlocal_qubit_masks[permutated_nonlocal_qubit_index_pairs[index].second] >> index)
                    - StateInteger{1u};
              nonlocal_qubit_value_masks[num_qubits_of_operation] = compl StateInteger{0u};

              std::transform(
                nonlocal_qubit_value_masks.rbegin(), std::prev(nonlocal_qubit_value_masks.rend()),
                std::next(nonlocal_qubit_value_masks.rbegin()), nonlocal_qubit_value_masks.rbegin(),
                std::minus<StateInteger>{});

              auto const last_nonlocal_qubit_value_wo_qubits
                = ::ket::utility::integer_exp2<StateInteger>(
                    mpi_policy.num_unit_qubits()
                    + ::ket::mpi::utility::policy::num_global_qubits(mpi_policy, communicator, environment)
                    - num_qubits_of_operation);
              for (auto nonlocal_qubit_value_wo_qubits = StateInteger{0u};
                   nonlocal_qubit_value_wo_qubits < last_nonlocal_qubit_value_wo_qubits;
                   ++nonlocal_qubit_value_wo_qubits)
              {
                auto nonlocal_qubit_value_base = StateInteger{0u};
                for (auto index = std::size_t{0u}; index < num_qubits_of_operation + std::size_t{1u}; ++index)
                  nonlocal_qubit_value_base
                    |= (nonlocal_qubit_value_wo_qubits bitand nonlocal_qubit_value_masks[index]) << index;

                auto const last_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
                for (auto qubit_mask1 = StateInteger{0u};
                     qubit_mask1 < last_qubit_mask - StateInteger{1u}; ++qubit_mask1)
                {
                  auto nonlocal_qubit_value1 = nonlocal_qubit_value_base;
                  for (auto index = BitInteger{0u}; index < num_qubits_of_operation; ++index)
                    nonlocal_qubit_value1
                      |= ((qubit_mask1 bitand (StateInteger{1u} << index)) >> index)
                         << (permutated_nonlocal_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second]
                             - num_local_qubits);

                  auto const unit_qubit_value1
                    = nonlocal_qubit_value1
                      bitand (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy)
                              - StateInteger{1u});
                  auto const rank_in_unit1
                    = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value1);
                  auto const global_qubit_value1
                    = (nonlocal_qubit_value1
                       bitand ((::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment)
                                - StateInteger{1u})
                               << mpi_policy.num_unit_qubits()))
                      >> mpi_policy.num_unit_qubits();
                  auto const rank1 = global_qubit_value1 * mpi_policy.num_processes_per_unit() + rank_in_unit1;

                  for (auto qubit_mask2 = qubit_mask1 + StateInteger{1u};
                       qubit_mask2 < last_qubit_mask; ++qubit_mask2)
                  {
                    auto nonlocal_qubit_value2 = nonlocal_qubit_value_base;
                    for (auto index = BitInteger{0u}; index < num_qubits_of_operation; ++index)
                      nonlocal_qubit_value2
                        |= ((qubit_mask2 bitand (StateInteger{1u} << index)) >> index)
                           << (permutated_nonlocal_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second]
                               - num_local_qubits);

                    auto const unit_qubit_value2
                      = nonlocal_qubit_value2
                        bitand (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy)
                                - StateInteger{1u});
                    auto const rank_in_unit2
                      = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value2);
                    auto const global_qubit_value2
                      = (nonlocal_qubit_value2
                         bitand ((::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment)
                                  - StateInteger{1u})
                                 << mpi_policy.num_unit_qubits()))
                        >> mpi_policy.num_unit_qubits();
                    auto const rank2 = global_qubit_value2 * mpi_policy.num_processes_per_unit() + rank_in_unit2;

                    if (rank2 == present_rank)
                    {
                      auto local_first_index2 = StateInteger{0u};
                      for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                        local_first_index2
                          |= ((qubit_mask1 bitand (StateInteger{1u} << index)) >> index)
                             << permutated_local_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second];

                      if (rank1 == present_rank)
                      {
                        auto local_first_index1 = StateInteger{0u};
                        for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                          local_first_index1
                            |= ((qubit_mask2 bitand (StateInteger{1u} << index)) >> index)
                               << permutated_local_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second];

                        auto const local_last_index1
                          = local_first_index1 + (data_block_size >> num_qubits_of_operation);
                        auto const data_block_index1
                          = ::ket::mpi::utility::policy::data_block_index(mpi_policy, unit_qubit_value1);
                        auto const data_block_index2
                          = ::ket::mpi::utility::policy::data_block_index(mpi_policy, unit_qubit_value2);

                        ::ket::mpi::utility::log_with_time_guard<char> print{"swap_local_data", environment};

                        ::ket::mpi::utility::detail::swap_local_data(
                          local_state,
                          data_block_index1, local_first_index1, local_last_index1,
                          data_block_index2, local_first_index2, data_block_size);
                      }
                      else // rank1 != present_rank
                      {
                        auto const data_block_index2
                          = ::ket::mpi::utility::policy::data_block_index(mpi_policy, unit_qubit_value2);
                        auto const local_last_index2
                          = local_first_index2 + (data_block_size >> num_qubits_of_operation);

                        ::ket::mpi::utility::log_with_time_guard<char> print{
                          ::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, ">::swap"),
                          environment};

                        interchange_qubits(
                          local_state, data_block_index2, data_block_size, local_first_index2, local_last_index2,
                          rank1, communicator, environment);
                      }
                    }
                    else if (rank1 == present_rank) // rank2 != present_rank
                    {
                      auto local_first_index1 = StateInteger{0u};
                      for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                        local_first_index1
                          |= ((qubit_mask2 bitand (StateInteger{1u} << index)) >> index)
                             << permutated_local_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second];

                      auto const data_block_index1
                        = ::ket::mpi::utility::policy::data_block_index(mpi_policy, unit_qubit_value1);
                      auto const local_last_index1
                        = local_first_index1 + (data_block_size >> num_qubits_of_operation);

                      ::ket::mpi::utility::log_with_time_guard<char> print{
                        ::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, ">::swap"),
                        environment};

                      interchange_qubits(
                        local_state, data_block_index1, data_block_size, local_first_index1, local_last_index1,
                        rank2, communicator, environment);
                    }
                  }
                }
              }
            }

            using ::ket::mpi::permutate;
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutate(permutation, qubits[index], local_swap_qubits[index]);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              std::begin(qubits) + new_unswappable_qubit_index + 1u, std::end(qubits),
              std::copy(
                std::begin(qubits), std::begin(qubits) + new_unswappable_qubit_index,
                std::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              std::begin(unswappable_qubits), std::end(unswappable_qubits),
              std::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>;
            lower_maybe_interchange_qubits::call(
              mpi_policy, parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, communicator, environment);
          }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            std::copy(
              std::begin(qubits) + new_unswappable_qubit_index + 1u, std::end(qubits),
              std::copy(
                std::begin(qubits), std::begin(qubits) + new_unswappable_qubit_index,
                std::begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(
              std::begin(unswappable_qubits), std::end(unswappable_qubits),
              std::begin(new_unswappable_qubits));
            new_unswappable_qubits.back() = qubits[new_unswappable_qubit_index];

            using lower_maybe_interchange_qubits
              = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  num_qubits_of_operation - 1u, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>;
            lower_maybe_interchange_qubits::call(
              mpi_policy, parallel_policy, local_state, new_qubits, new_unswappable_qubits,
              permutation, buffer, datatype, communicator, environment);
          }
        }; // struct maybe_interchange_qubits<num_qubits_of_operation, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>&,
            yampi::communicator const&, yampi::environment const&)
          { }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>&,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const&, yampi::environment const&)
          { }
        }; // struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>

        template <typename MpiPolicy>
        struct rank_index_to_qubit_value;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        {
          template <typename LocalState>
          static StateInteger call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index)
          {
            // g
            auto const global_qubit_value
              = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, rank);
            // r_u
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, rank);
            // 2^L
            auto const data_block_size
              = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit);
            // i_u = i / 2^L
            auto const data_block_index = index / data_block_size;
            // l = i % 2^L
            auto const local_qubit_value = index % data_block_size;
            // u
            auto const unit_qubit_value
              = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit);

            return
              global_qubit_value * ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy) * data_block_size
              + unit_qubit_value * data_block_size + local_qubit_value;
          }
        }; // struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >

        template <typename MpiPolicy>
        struct qubit_value_to_rank_index;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        {
          template <typename LocalState>
          static std::pair<yampi::rank, StateInteger> call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState const& local_state, StateInteger const qubit_value,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            // 2^L
            auto const data_block_size
              = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
            // g
            auto const global_qubit_value
              = qubit_value / (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy) * data_block_size);
            auto const nonglobal_qubit_value
              = qubit_value % (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy) * data_block_size);
            // u
            auto const unit_qubit_value = nonglobal_qubit_value / data_block_size;
            // l
            auto const local_qubit_value = nonglobal_qubit_value % data_block_size;
            // r_u
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value);
            // i_u
            auto const data_block_index = ::ket::mpi::utility::policy::data_block_index(mpi_policy, unit_qubit_value);

            return std::make_pair(
              ::ket::mpi::utility::policy::rank(mpi_policy, global_qubit_value, rank_in_unit),
              data_block_index * data_block_size + local_qubit_value);
          }
        }; // struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >

# ifdef KET_USE_DIAGONAL_LOOP
        template <typename MpiPolicy>
        struct diagonal_loop;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        struct diagonal_loop< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        {
          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits... control_qubits)
          {
            using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
            auto unit_permutated_control_qubits = std::array<permutated_control_qubit_type, 0u>{};
            auto local_permutated_control_qubits = std::array<permutated_control_qubit_type, 0u>{};

            auto const present_rank = communicator.rank(environment);
            auto const present_rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, present_rank);
            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_unit_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, present_rank_in_unit)};
            auto const least_global_permutated_qubit = least_unit_permutated_qubit + mpi_policy.num_unit_qubits();

            call_impl(
              mpi_policy, parallel_policy, local_state, permutation,
              present_rank, present_rank_in_unit,
              least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
              std::forward<Function0>(function0),
              std::forward<Function1>(function1),
              unit_permutated_control_qubits, local_permutated_control_qubits,
              control_qubits...);
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits,
            typename... ControlQubits>
          static void call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_local_control_qubits > const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits... control_qubits)
          {
            auto const permutated_control_qubit = permutation[control_qubit];

            if (permutated_control_qubit < least_unit_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_local_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_local_control_qubits + 1u>{};
              std::copy(
                std::begin(local_permutated_control_qubits),
                std::end(local_permutated_control_qubits),
                std::begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, local_state, permutation,
                present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0),
                std::forward<Function1>(function1),
                unit_permutated_control_qubits, new_local_permutated_control_qubits,
                control_qubits...);
            }
            else if (permutated_control_qubit < least_global_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_unit_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_unit_control_qubits + 1u>{};
              std::copy(
                std::begin(unit_permutated_control_qubits),
                std::end(unit_permutated_control_qubits),
                std::begin(new_unit_permutated_control_qubits));
              new_unit_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, local_state, permutation,
                present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0),
                std::forward<Function1>(function1),
                new_unit_permutated_control_qubits, local_permutated_control_qubits,
                control_qubits...);
            }
            else
            {
              static constexpr auto zero_state_integer = StateInteger{0u};
              static constexpr auto one_state_integer = StateInteger{1u};

              auto const mask
                = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand mask) != zero_state_integer)
                call_impl(
                  mpi_policy, parallel_policy, local_state, permutation,
                  present_rank, present_rank_in_unit,
                  least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                  std::forward<Function0>(function0),
                  std::forward<Function1>(function1),
                  unit_permutated_control_qubits, local_permutated_control_qubits,
                  control_qubits...);
            }
          }

          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits>
          static void call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array<
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >,
              num_local_control_qubits > const& local_permutated_control_qubits)
          {
            auto const permutated_target_qubit = permutation[target_qubit];

            static constexpr auto zero_state_integer = StateInteger{0u};
            static constexpr auto one_state_integer = StateInteger{1u};

            auto const num_data_blocks
              = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, present_rank_in_unit);
            auto const data_block_size
              = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, present_rank_in_unit);

            auto const last_local_qubit_value = one_state_integer << least_unit_permutated_qubit;

            if (permutated_target_qubit < least_unit_permutated_qubit)
            {
              auto const target_mask = one_state_integer << permutated_target_qubit;

              for (auto data_block_index = StateInteger{0u};
                   data_block_index < num_data_blocks; ++data_block_index)
              {
                auto const unit_qubit_value
                  = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                if (std::any_of(
                      std::begin(unit_permutated_control_qubits), std::end(unit_permutated_control_qubits),
                      [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                      {
                        return
                          (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                            == zero_state_integer;
                      }))
                  continue;

#   ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
                ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                  parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                  [&function0, &function1, target_mask](auto const iter, StateInteger const state_integer)
                  {
                    if ((state_integer bitand target_mask) == zero_state_integer)
                      function0(iter, state_integer);
                    else
                      function1(iter, state_integer);
                  });
#   else // BOOST_NO_CXX14_GENERIC_LAMBDAS
                ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                  parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                  make_call_function_if_local(
                    std::forward<Function0>(function0), std::forward<Function1>(function1), target_mask));
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
              }
            }
            else if (permutated_target_qubit < least_global_permutated_qubit)
            {
              auto const target_mask = one_state_integer << (permutated_target_qubit - least_unit_permutated_qubit);

              for (auto data_block_index = StateInteger{0u};
                   data_block_index < num_data_blocks; ++data_block_index)
              {
                auto const unit_qubit_value
                  = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                if (std::any_of(
                      std::begin(unit_permutated_control_qubits), std::end(unit_permutated_control_qubits),
                      [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                      {
                        return
                          (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                            == zero_state_integer;
                      }))
                  continue;

                if ((unit_qubit_value bitand target_mask) == zero_state_integer)
                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function0>(function0));
                else
                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function1>(function1));
              }
            }
            else
            {
              auto const target_mask = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask)
                  == zero_state_integer)
                for (auto data_block_index = StateInteger{0u};
                     data_block_index < num_data_blocks; ++data_block_index)
                {
                  auto const unit_qubit_value
                    = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                  using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                  if (std::any_of(
                        std::begin(unit_permutated_control_qubits), std::end(unit_permutated_control_qubits),
                        [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                        {
                          return
                            (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                              == zero_state_integer;
                        }))
                    continue;

                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function0>(function0));
                }
              else
                for (auto data_block_index = StateInteger{0u};
                     data_block_index < num_data_blocks; ++data_block_index)
                {
                  auto const unit_qubit_value
                    = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                  using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                  if (std::any_of(
                        std::begin(unit_permutated_control_qubits), std::end(unit_permutated_control_qubits),
                        [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                        {
                          return
                            (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                              == zero_state_integer;
                        }))
                    continue;

                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, local_state, data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function1>(function1));
                }
            }
          }

#   ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Function0, typename Function1>
          struct call_function_if_local
          {
            Function0 function0_;
            Function1 function1_;
            StateInteger target_mask_;

            call_function_if_local(Function0&& function0, Function1&& function1, StateInteger const target_mask)
              : function0_{std::forward<Function0>(function0)},
                function1_{std::forward<Function1>(function1)},
                target_mask_{target_mask}
            { }

            template <typename Iterator>
            void operator()(Iterator const iter, StateInteger const state_integer)
            {
              static constexpr auto zero_state_integer = StateInteger{0u};

              if ((state_integer bitand target_mask_) == zero_state_integer)
                function0_(iter, state_integer);
              else
                function1_(iter, state_integer);
            }
          }; // struct call_function_if_local<Function0, Function1, StateInteger>

          template <typename Function0, typename Function1>
          static call_function_if_local<Function0, Function1>
          make_call_function_if_local(Function0&& function0, Function1&& function1, StateInteger const target_mask)
          {
            return call_function_if_local<Function0, Function1>{
              std::forward<Function0>(function0), std::forward<Function1>(function1), target_mask};
          }
#   endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }; // struct diagonal_loop< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
# endif // KET_USE_DIAGONAL_LOOP
      } // namespace dispatch

      template <
        typename StateInteger, typename BitInteger, typename NumProcesses,
        typename ParallelPolicy, typename LocalState, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        using maybe_interchange_qubits_impl
          = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
              num_qubits_of_operation,
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>;
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto unswappable_qubits = std::array<qubit_type, 0u>{};

        maybe_interchange_qubits_impl::call(
          mpi_policy, parallel_policy,
          local_state, qubits, unswappable_qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename StateInteger, typename BitInteger, typename NumProcesses,
        typename ParallelPolicy, typename LocalState, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        using maybe_interchange_qubits_impl
          = ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
              num_qubits_of_operation,
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>;
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;

        auto unswappable_qubits = std::array<qubit_type, 0u>{};

        maybe_interchange_qubits_impl::call(
          mpi_policy, parallel_policy,
          local_state, qubits, unswappable_qubits, permutation,
          buffer, datatype, communicator, environment);
      }

      template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
      inline StateInteger rank_index_to_qubit_value(
        ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
        LocalState const& local_state, yampi::rank const rank,
        StateInteger const index)
      {
        return ::ket::mpi::utility::dispatch::rank_index_to_qubit_value<
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(
            mpi_policy, local_state, rank, index);
      }

      template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
      inline std::pair<yampi::rank, StateInteger> qubit_value_to_rank_index(
        ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
        LocalState const& local_state, StateInteger const qubit_value,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::utility::dispatch::qubit_value_to_rank_index<
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(
            mpi_policy, local_state, qubit_value, communicator, environment);
      }

# ifdef KET_USE_DIAGONAL_LOOP
      template <
        typename StateInteger, typename BitInteger, typename NumProcesses,
        typename ParallelPolicy, typename LocalState, typename Allocator,
        typename Function0, typename Function1, typename... ControlQubits>
      inline void diagonal_loop(
        ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        Function0&& function0, Function1&& function1,
        ControlQubits... control_qubits)
      {
        assert(not ::ket::mpi::page::is_on_page(permutation[target_qubit], local_state));
        assert(::ket::mpi::page::are_on_nonpage(local_state, permutation, control_qubits...));

        return ::ket::mpi::utility::dispatch::diagonal_loop<
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
            target_qubit,
            std::forward<Function0>(function0), std::forward<Function1>(function1),
            control_qubits...);
      }
# endif // KET_USE_DIAGONAL_LOOP
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_UNIT_MPI_HPP
