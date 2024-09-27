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

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# ifndef NDEBUG
#   include <yampi/lowest_io_process.hpp>
# endif
# ifdef KET_USE_BARRIER
#   include <yampi/barrier.hpp>
# endif // KET_USE_BARRIER
# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
#   include <yampi/noncontiguous_buffer.hpp>
#   include <yampi/noncontiguous_complete_exchange.hpp>
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# ifndef NDEBUG
#   include <ket/mpi/page/is_on_page.hpp>
# endif
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/detail/make_local_swap_qubit.hpp>
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
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        class unit_mpi;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> yampi::rank;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto num_data_blocks(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::rank const rank_in_unit)
        -> StateInteger;

        /*
         * qubit index: xxxxx|xxxxxx|xxxxxxxxx, global qubits, unit qubits, and local qubits from left to right
         * N = L + K + M: the number of qubits
         * L: the number of local qubits, l: value of local qubits
         * K: the number of unit qubits, u: value of unit qubits
         * M: the number of global qubits, g: value of global qubits
         * Each unit has n_u MPI processes, and the value of global qubits g is unit index.
         * The total number of MPI processes is 2^M n_u.
         *
         * The number of data blocks k~ in the MPI process with rank r is k~ = k+1 if 0 <= r_u < m, k if m <= r_u < n_u, where r_u = r % n_u is "rank in unit", and 0 < m < n_u is a constant.
         * The total number of data blocks in each unit should be equal to 2^K.
         * Therefore, 2^K = m (k+1) + (n_u-m) k = k n_u + m does hold.
         * This means k and m are determined as k = 2^K / n_u and m = 2^K % n_u.
         * Note that '/' is integer division.
         *
         * The rank in unit r_u is also determined by r_u = u / (k+1) if 0 <= u < m(k+1), m + [u - m(k+1)] / k if m(k+1) <= u < 2^K.
         * Actual rank of the MPI process is given by r = g n_u + r_u.
         * Moreover, element index in the MPI process is given by i = i_u * 2^L + l, where i_u is an index of a data block in the MPI process and satisfies i_u = u % (k+1) if 0 <= u < m(k+1), [u - m(k+1)] % k if m(k+1) <= u < 2^K.
         * Note that u = (k+1) r_u + i_u if 0 <= r_u < m, k r_u + i_u + m if m <= r_u < n_u.
         */
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        class unit_mpi
        {
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
          static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
          static_assert(std::is_unsigned<NumProcesses>::value, "NumProcesses should be unsigned");

          BitInteger num_unit_qubits_; // K
          NumProcesses num_processes_per_unit_; // n_u

          StateInteger num_data_blocks_in_process_b_; // k, num_data_blocks_in_process_a_ == num_data_blocks_in_process_b_ + 1;
          NumProcesses num_processes_a_per_unit_; // m, num_processes_b_per_unit_ == num_processes_per_unit_ - num_processes_a_per_unit_;

         public:
          unit_mpi(BitInteger const num_unit_qubits, NumProcesses const num_processes_per_unit)
            : num_unit_qubits_{num_unit_qubits},
              num_processes_per_unit_{num_processes_per_unit},
              num_data_blocks_in_process_b_{
                ::ket::utility::integer_exp2<StateInteger>(num_unit_qubits) / static_cast<StateInteger>(num_processes_per_unit)},
              num_processes_a_per_unit_{
                ::ket::utility::integer_exp2<NumProcesses>(num_unit_qubits) % num_processes_per_unit}
          {
            assert(num_unit_qubits >= BitInteger{1u});
            assert(
              num_processes_per_unit >= NumProcesses{1u}
              and num_processes_per_unit <= ::ket::utility::integer_exp2<NumProcesses>(num_unit_qubits));
          }

          // K
          auto num_unit_qubits() const noexcept -> BitInteger const& { return num_unit_qubits_; }
          // n_u
          auto num_processes_per_unit() const noexcept -> NumProcesses const& { return num_processes_per_unit_; }

          // k
          auto num_data_blocks_in_process_b() const noexcept -> StateInteger const& { return num_data_blocks_in_process_b_; }
          // m
          auto num_processes_a_per_unit() const noexcept -> NumProcesses const& { return num_processes_a_per_unit_; }
        }; // class unit_mpi<StateInteger, BitInteger, NumProcesses>

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto make_unit_mpi(BitInteger const num_unit_qubits, NumProcesses const num_unit_processes) noexcept
        -> ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>
        { return {num_unit_qubits, num_unit_processes}; }

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
        inline auto num_unit_qubit_values(::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy)
        -> StateInteger
        { return ::ket::utility::integer_exp2<StateInteger>(mpi_policy.num_unit_qubits()); }

        // r_u = r % n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto rank_in_unit(::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy, yampi::rank const rank)
        -> yampi::rank
        {
          assert(rank.mpi_rank() >= 0);
          return rank % mpi_policy.num_processes_per_unit();
        }

        // r_u = r % n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> yampi::rank
        { return ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator.rank(environment)); }

        // r_u = u / (k+1) if 0 <= u < m(k+1), m + [u - m(k+1)] / k if m(k+1) <= u < 2^K
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto rank_in_unit(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const unit_qubit_value)
        -> yampi::rank
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));

          // k+1
          auto const num_data_blocks_in_process_a = mpi_policy.num_data_blocks_in_process_b() + StateInteger{1u};
          // m(k+1)
          auto const threshold = static_cast<StateInteger>(mpi_policy.num_processes_a_per_unit()) * num_data_blocks_in_process_a;

          auto const result
            = unit_qubit_value < threshold
              ? static_cast<int>(unit_qubit_value / num_data_blocks_in_process_a)
              : static_cast<int>(mpi_policy.num_processes_a_per_unit() + (unit_qubit_value - threshold) / mpi_policy.num_data_blocks_in_process_b());

          assert(result < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return yampi::rank{result};
        }

        // i_u = u % (k+1) if 0 <= u < m(k+1), [u - m(k+1)] % k if m(k+1) <= u < 2^K
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto data_block_index(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const unit_qubit_value)
        -> StateInteger
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));

          // k+1
          auto const num_data_blocks_in_process_a = mpi_policy.num_data_blocks_in_process_b() + StateInteger{1u};
          // m(k+1)
          auto const threshold = static_cast<StateInteger>(mpi_policy.num_processes_a_per_unit()) * num_data_blocks_in_process_a;

          return unit_qubit_value < threshold
            ? static_cast<int>(unit_qubit_value % num_data_blocks_in_process_a)
            : static_cast<int>((unit_qubit_value - threshold) % mpi_policy.num_data_blocks_in_process_b());
        }

        // k~ = k+1 if 0 <= r_u < m, k if m <= r_u < n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto num_data_blocks(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::rank const rank_in_unit)
        -> StateInteger
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));

          return rank_in_unit < yampi::rank{static_cast<int>(mpi_policy.num_processes_a_per_unit())}
            ? mpi_policy.num_data_blocks_in_process_b() + StateInteger{1u}
            : mpi_policy.num_data_blocks_in_process_b();
        }

        namespace dispatch
        {
          // k~
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct num_data_blocks< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> StateInteger
            {
              return ::ket::mpi::utility::policy::num_data_blocks(
                mpi_policy, ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
            }
          }; // struct num_data_blocks< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        // u = (k+1) r_u + i_u if 0 <= r_u < m, k r_u + i_u + m if m <= r_u < n_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto unit_qubit_value(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const data_block_index, yampi::rank const rank_in_unit)
        -> StateInteger
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          assert(data_block_index < ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

          auto const result
            = rank_in_unit < yampi::rank{static_cast<int>(mpi_policy.num_processes_a_per_unit())}
              ? static_cast<StateInteger>((mpi_policy.num_data_blocks_in_process_b() + NumProcesses{1u}) * rank_in_unit.mpi_rank()) + data_block_index
              : static_cast<StateInteger>(mpi_policy.num_data_blocks_in_process_b() * rank_in_unit.mpi_rank() + mpi_policy.num_processes_a_per_unit()) + data_block_index;

          assert(result < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return result;
        }

        // 2^M
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto num_units(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> StateInteger
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
        inline auto num_global_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
          StateInteger const num_units)
        -> StateInteger
        {
          assert(num_units >= StateInteger{1u});
          auto const result = ::ket::utility::integer_log2<BitInteger>(num_units);
          assert(::ket::utility::integer_exp2<StateInteger>(result) == num_units);
          return result;
        }

        namespace dispatch
        {
          // M
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct num_global_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> StateInteger
            {
              return ::ket::mpi::utility::policy::num_global_qubits(
                mpi_policy, ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
            }
          }; // struct num_global_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        namespace dispatch
        {
          // g = r / n_u
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct global_qubit_value< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              yampi::rank const rank)
            -> StateInteger
            {
              assert(rank.mpi_rank() >= 0);
              return
                static_cast<StateInteger>(rank.mpi_rank())
                / static_cast<StateInteger>(mpi_policy.num_processes_per_unit());
            }

            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> StateInteger
            {
              auto const result = call(mpi_policy, communicator.rank(environment));
              assert(result < ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
              return result;
            }
          }; // struct global_qubit_value< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        // r = g n_u + r_u
        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        inline auto rank(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          StateInteger const global_qubit_value, yampi::rank const rank_in_unit)
        -> yampi::rank
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return global_qubit_value * mpi_policy.num_processes_per_unit() + rank_in_unit;
        }

        namespace dispatch
        {
          // r = g n_u + r_u
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct rank< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              StateInteger const global_qubit_value,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> yampi::rank
            {
              assert(global_qubit_value < ::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment));
              return ::ket::mpi::utility::policy::rank(
                mpi_policy, global_qubit_value,
                ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
            }
          }; // struct rank< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        // 2^L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline auto data_block_size(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, yampi::rank const rank_in_unit)
        -> StateInteger
        {
          using std::begin;
          using std::end;
          auto const local_state_size = static_cast<StateInteger>(std::distance(begin(local_state), end(local_state)));
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          assert(local_state_size % ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit) == 0u);

          auto const result
            = static_cast<StateInteger>(local_state_size / ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));
          assert(::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(result)) == result);
          return result;
        }

        namespace dispatch
        {
          // 2^L
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct data_block_size< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            template <typename LocalState>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> StateInteger
            {
              return ::ket::mpi::utility::policy::data_block_size(
                mpi_policy, local_state,
                ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment));
            }
          }; // struct data_block_size< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        // 2^L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline auto data_block_size(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, StateInteger const unit_qubit_value)
        -> StateInteger
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return ::ket::mpi::utility::policy::data_block_size(
            mpi_policy, local_state,
            ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value));
        }

        namespace dispatch
        {
          // L
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct num_local_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              StateInteger const data_block_size)
              -> BitInteger
            {
              assert(data_block_size >= StateInteger{2u});
              auto const result = ::ket::utility::integer_log2<BitInteger>(data_block_size);
              assert(::ket::utility::integer_exp2<StateInteger>(result) == data_block_size);
              return result;
            }

            template <typename LocalState>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> BitInteger
            {
              return call(
                mpi_policy,
                ::ket::mpi::utility::policy::dispatch::data_block_size< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(
                  mpi_policy, local_state, communicator, environment));
            }
          }; // struct num_local_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline auto num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, yampi::rank const rank_in_unit)
        -> BitInteger
        {
          assert(rank_in_unit.mpi_rank() >= 0 and rank_in_unit.mpi_rank() < static_cast<int>(mpi_policy.num_processes_per_unit()));
          return ::ket::mpi::utility::policy::num_local_qubits(
            mpi_policy, ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit));
        }

        // L
        template <typename StateInteger, typename BitInteger, typename NumProcesses, typename LocalState>
        inline auto num_local_qubits(
          ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
          LocalState const& local_state, StateInteger const unit_qubit_value)
        -> BitInteger
        {
          assert(unit_qubit_value < ::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy));
          return ::ket::mpi::utility::policy::num_local_qubits(
            mpi_policy, ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, unit_qubit_value));
        }

        namespace dispatch
        {
          // N = L + K + M
          template <typename StateInteger, typename BitInteger, typename NumProcesses>
          struct num_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
          {
            template <typename LocalState>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState const& local_state,
              yampi::communicator const& communicator, yampi::environment const& environment)
            -> BitInteger
            {
              return ::ket::mpi::utility::policy::dispatch::num_local_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(mpi_policy, local_state, communicator, environment)
                + mpi_policy.num_unit_qubits()
                + ::ket::mpi::utility::policy::dispatch::num_global_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >::call(mpi_policy, communicator, environment);
            }
          }; // struct num_qubits< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        } // namespace dispatch
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
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_unit_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)};

            auto permutated_nonlocal_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_nonlocal_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_nonlocal_swap_qubits[index] >= least_unit_permutated_qubit)
                continue;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, communicator, environment);
              return;
            }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
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
              },
              yampi::predefined_datatype< ::ket::utility::meta::range_value_t<LocalState> >{},
              [](
                LocalState& local_state,
                std::vector<int> const& counts, std::vector<int> const& displacements, std::vector<yampi::datatype> const& datatypes,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                assert(counts.size() == displacements.size() and counts.size() == datatypes.size() and static_cast<int>(counts.size()) == communicator.size(environment));

                using std::begin;
                yampi::noncontiguous_complete_exchange(
                  yampi::in_place,
                  yampi::make_noncontiguous_buffer(begin(local_state), begin(counts), begin(displacements), begin(datatypes)),
                  communicator, environment);
              });
# else // KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
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
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS
          }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_unit_permutated_qubit
              = permutated_qubit_type{::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, communicator, environment)};

            auto permutated_nonlocal_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              permutated_nonlocal_swap_qubits[index] = permutation[qubits[index]];

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              if (permutated_nonlocal_swap_qubits[index] >= least_unit_permutated_qubit)
                continue;

              call_lower_maybe_interchange_qubits(
                index,
                mpi_policy, parallel_policy, local_state, qubits, unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);
              return;
            }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
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
              },
              datatype,
              [](
                LocalState& local_state,
                std::vector<int> const& counts, std::vector<int> const& displacements, std::vector<yampi::datatype> const& datatypes,
                yampi::communicator const& communicator, yampi::environment const& environment)
              {
                assert(counts.size() == displacements.size() and counts.size() == datatypes.size() and static_cast<int>(counts.size()) == communicator.size(environment));

                using std::begin;
                yampi::noncontiguous_complete_exchange(
                  yampi::in_place,
                  yampi::make_noncontiguous_buffer(begin(local_state), begin(counts), begin(displacements), begin(datatypes)),
                  communicator, environment);
              });
# else // KET_USE_COLLECTIVE_COMMUNICATIONS
            do_call(
              mpi_policy, parallel_policy, local_state,
              least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              qubits, unswappable_qubits, permutation, communicator, environment,
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
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS
          }

         private:
          template <typename ParallelPolicy, typename LocalState, std::size_t num_unswappable_qubits, typename Allocator>
          static auto initialize_local_swap_qubits(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation >& local_swap_qubits,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation >& permutated_local_swap_qubits,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
# ifdef KET_USE_BARRIER
            ::yampi::barrier(communicator, environment);
# endif // KET_USE_BARRIER

            ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, '>'), environment};

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator{yampi::tags::world_communicator}.rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);

            // (ex.: num_qubits_of_operation == 3)
            //  Swaps between xxbxb'x|b''xx|cc'c''xxxxxxxx and
            // xxcxc'x|c''xx|bb'b''xxxxxxxx (c = b or ~b). Upper, middle, and
            // lower qubits and are global, unit, and local qubits,
            // respectively. The first three upper qubits in the local qubits
            // are "local swap qubits". Three bits in global qubits and the
            // "local swap qubits" would be swapped.

            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
            {
              permutated_local_swap_qubits[index] = least_unit_permutated_qubit - static_cast<BitInteger>(std::size_t{1u} + index);
              local_swap_qubits[index]
                = ::ket::mpi::utility::detail::make_local_swap_qubit(
                    parallel_policy, local_state, permutation,
                    unswappable_qubits, permutated_local_swap_qubits[index],
                    num_data_blocks, data_block_size, communicator, environment);
            }

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <typename Allocator>
          static auto update_permutation(
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& local_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
              ::ket::mpi::permutate(permutation, qubits[index], local_swap_qubits[index]);

# ifndef NDEBUG
            auto const maybe_io_rank = yampi::lowest_io_process(environment);
            auto const my_rank = yampi::communicator{yampi::tags::world_communicator}.rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG

# ifdef KET_USE_BARRIER
            ::yampi::barrier(communicator, environment);
# endif // KET_USE_BARRIER
          }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <typename PermutatedQubitIterator>
          static auto generate_global_key(
            StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            PermutatedQubitIterator const permutated_global_swap_qubits_first, PermutatedQubitIterator const permutated_global_swap_qubits_last)
          -> StateInteger
          {
            auto const num_global_swap_qubits = permutated_global_swap_qubits_last - permutated_global_swap_qubits_first;

            // global_key == zyx for global_qubit_value == *****y****x****z**** (permutated_global_swap_qubits[0]==permutated_qubit{9}, permutated_global_swap_qubits[1]==permutated_qubit{14}, permutated_global_swap_qubits[2]==permutated_qubit{4})
            auto result = StateInteger{0u};
            for (auto index = decltype(num_global_swap_qubits){0}; index < num_global_swap_qubits; ++index)
            {
              auto const shift = *(permutated_global_swap_qubits_first + index) - least_global_permutated_qubit;
              result |= ((global_qubit_value bitand (StateInteger{1u} << shift)) >> shift) << index;
            }

            return result;
          }

          template <typename PermutatedQubitIterator>
          static auto generate_color_key(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            PermutatedQubitIterator const permutated_global_swap_qubits_first, PermutatedQubitIterator const permutated_global_swap_qubits_last,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> std::pair<yampi::color, int>
          {
            auto const num_global_swap_qubits = permutated_global_swap_qubits_last - permutated_global_swap_qubits_first;

            // initialization of permutated_global_qubit_index_pairs ({sorted_global_qubit, corresponding_index_in_some_arrays}, ...)
            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            using permutated_global_qubit_index_pair_type = std::pair<permutated_qubit_type, std::size_t>;
            auto permutated_global_qubit_index_pairs
              = std::vector<permutated_global_qubit_index_pair_type>(num_global_swap_qubits);
            for (auto index = decltype(num_global_swap_qubits){0u}; index < num_global_swap_qubits; ++index)
              permutated_global_qubit_index_pairs[index] = std::make_pair(*(permutated_global_swap_qubits_first + index), index);

            using std::begin;
            using std::end;
            std::sort(
              begin(permutated_global_qubit_index_pairs), end(permutated_global_qubit_index_pairs),
              [](permutated_global_qubit_index_pair_type const& lhs, permutated_global_qubit_index_pair_type const& rhs)
              { return lhs.first < rhs.first;});

            // initialization of permutated_global_qubit_masks (000001000000, 000000001000, 001000000000)
            auto permutated_global_qubit_masks = std::vector<StateInteger>(num_global_swap_qubits);
            for (auto index = decltype(num_global_swap_qubits){0u}; index < num_global_swap_qubits; ++index)
              permutated_global_qubit_masks[index] = (StateInteger{1u} << *(permutated_global_swap_qubits_first + index)) >> least_global_permutated_qubit;

            // initialization of global_qubit_value_masks (000000111, 000011000, 001100000, 110000000)
            auto global_qubit_value_masks = std::vector<StateInteger>(num_global_swap_qubits + decltype(num_global_swap_qubits){1u});
            for (auto index = decltype(num_global_swap_qubits){0u}; index < num_global_swap_qubits; ++index)
              global_qubit_value_masks[index] = (permutated_global_qubit_masks[permutated_global_qubit_index_pairs[index].second] >> index) - StateInteger{1u};
            global_qubit_value_masks[num_global_swap_qubits] = compl StateInteger{0u};

            using std::rbegin;
            using std::rend;
            std::transform(
              rbegin(global_qubit_value_masks), std::prev(rend(global_qubit_value_masks)),
              std::next(rbegin(global_qubit_value_masks)), rbegin(global_qubit_value_masks),
              std::minus<StateInteger>{});

            auto color_integer = StateInteger{0u};
            for (auto index = decltype(num_global_swap_qubits){0u}; index <= num_global_swap_qubits; ++index)
              color_integer |= ((global_qubit_value >> index) bitand global_qubit_value_masks[index]);

            // local_rank == yampi::rank{global_key * mpi_policy.num_processes_per_unit()} + rank_in_unit;
            auto const global_key
              = generate_global_key(
                  global_qubit_value, least_global_permutated_qubit,
                  permutated_global_swap_qubits_first, permutated_global_swap_qubits_last);
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
            auto const key = global_key * static_cast<StateInteger>(mpi_policy.num_processes_per_unit()) + static_cast<StateInteger>(rank_in_unit.mpi_rank());

            return {yampi::color{static_cast<int>(color_integer)}, static_cast<int>(key)};
          }

          template <typename LocalState, typename DerivedDatatype, typename Function>
          static auto interchange_qubits_collective(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState& local_state, yampi::datatype_base<DerivedDatatype> const& datatype,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& noncontiguous_complete_exchange_qubits)
          -> void
          {
            auto const least_global_permutated_qubit = least_unit_permutated_qubit + mpi_policy.num_unit_qubits();
            auto permutated_unit_global_swap_qubits = permutated_nonlocal_swap_qubits;

            using std::begin;
            using std::end;
            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const permutated_global_swap_qubits_last = end(permutated_unit_global_swap_qubits);
            auto const permutated_global_swap_qubits_first
              = std::stable_partition(
                  begin(permutated_unit_global_swap_qubits), permutated_global_swap_qubits_last,
                  [least_global_permutated_qubit](permutated_qubit_type const permutated_qubit)
                  { return permutated_qubit < least_global_permutated_qubit; });

            auto const source_global_qubit_value = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment);
            auto const color_key
              = generate_color_key(
                  mpi_policy, source_global_qubit_value, least_global_permutated_qubit,
                  permutated_global_swap_qubits_first, permutated_global_swap_qubits_last,
                  communicator, environment);
            auto const local_communicator = yampi::communicator{communicator, color_key.first, color_key.second, environment};
            auto const num_local_processes = local_communicator.size(environment);
            auto const present_local_rank = local_communicator.rank(environment);

            // initialization of permutated_nonlocal_qubit_masks (000001000000, 000000001000, 001000000000)
            auto permutated_nonlocal_qubit_masks = std::array<StateInteger, num_qubits_of_operation>{};
            std::transform(
              begin(permutated_nonlocal_swap_qubits), end(permutated_nonlocal_swap_qubits),
              begin(permutated_nonlocal_qubit_masks),
              [least_unit_permutated_qubit](permutated_qubit_type const permutated_nonlocal_swap_qubit)
              { return (StateInteger{1u} << permutated_nonlocal_swap_qubit) >> least_unit_permutated_qubit; });

            constexpr auto last_swap_qubit_value = ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
            auto const source_rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
            auto const num_local_qubits = ::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, source_rank_in_unit);
            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, source_rank_in_unit);
            auto const num_chunks = num_data_blocks * last_swap_qubit_value;
            auto intraprocess_swap_chunk_index_pairs = std::vector<std::pair<decltype(num_chunks), decltype(num_chunks)>>{};
            intraprocess_swap_chunk_index_pairs.reserve(num_chunks);
            auto target_local_ranks = std::vector<yampi::rank>(num_chunks, yampi::null_process);
            auto num_transferring_chunks = 0;
            auto num_transferring_chunks_vec= std::vector<int>(num_local_processes);

            for (auto source_data_block_index = decltype(num_data_blocks){0u}; source_data_block_index < num_data_blocks; ++source_data_block_index)
            {
              // (xbxb'x|)xb''x(|xxxxxxxx)
              auto const source_unit_qubit_value
                = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, source_data_block_index, source_rank_in_unit);
              // xbxb'x|xb''x(|xxxxxxxx)
              auto const source_nonlocal_qubit_value
                = (source_global_qubit_value << (least_global_permutated_qubit - least_unit_permutated_qubit)) bitor source_unit_qubit_value;

              for (auto nonlocal_qubit_mask = StateInteger{1u}; nonlocal_qubit_mask < last_swap_qubit_value; ++nonlocal_qubit_mask)
              {
                // xcxc'x|xc''x(|xxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
                auto mask = StateInteger{0u};
                for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                  mask |= ((nonlocal_qubit_mask bitand (StateInteger{1u} << index)) >> index) << (permutated_nonlocal_swap_qubits[index] - least_unit_permutated_qubit);
                auto const target_nonlocal_qubit_value = source_nonlocal_qubit_value xor mask;

                auto const target_global_qubit_value
                  = target_nonlocal_qubit_value >> (least_global_permutated_qubit - least_unit_permutated_qubit);
                auto const target_unit_qubit_value
                  = target_nonlocal_qubit_value bitand (((StateInteger{1u} << (least_global_permutated_qubit - least_unit_permutated_qubit)) - StateInteger{1u}));
                auto const target_rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, target_unit_qubit_value);

                // (0000000|0000|)cc'c''(00000)
                auto source_local_swap_qubits_value = StateInteger{0u};
                for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                  source_local_swap_qubits_value
                    |= ((((target_nonlocal_qubit_value bitand permutated_nonlocal_qubit_masks[index])
                          << least_unit_permutated_qubit) >> permutated_nonlocal_swap_qubits[index]) << permutated_local_swap_qubits[index])
                       >> (num_local_qubits - num_qubits_of_operation);
                auto const source_chunk_index = source_data_block_index * last_swap_qubit_value + source_local_swap_qubits_value;

                // local_rank == yampi::rank{global_key * mpi_policy.num_processes_per_unit()} + rank_in_unit;
                auto const target_global_key
                  = generate_global_key(
                      target_global_qubit_value, least_global_permutated_qubit,
                      permutated_global_swap_qubits_first, permutated_global_swap_qubits_last);
                auto const target_local_rank
                  = target_global_key * mpi_policy.num_processes_per_unit() + target_rank_in_unit;

                if (target_local_rank == present_local_rank)
                {
                  auto const target_data_block_index = ::ket::mpi::utility::policy::data_block_index(mpi_policy, target_unit_qubit_value);

                  // (0000000|0000|)bb'b''(00000)
                  auto target_local_swap_qubits_value = StateInteger{0u};
                  for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                    target_local_swap_qubits_value
                      |= ((((source_nonlocal_qubit_value << least_unit_permutated_qubit) bitand permutated_nonlocal_qubit_masks[index])
                           >> permutated_nonlocal_swap_qubits[index]) << permutated_local_swap_qubits[index])
                         >> (num_local_qubits - num_qubits_of_operation);
                  auto const target_chunk_index = target_data_block_index * last_swap_qubit_value + target_local_swap_qubits_value;

                  if (source_chunk_index < target_chunk_index)
                    intraprocess_swap_chunk_index_pairs.push_back({source_chunk_index, target_chunk_index});
                }

                target_local_ranks[source_chunk_index] = target_local_rank;
                ++num_transferring_chunks;
                ++num_transferring_chunks_vec[target_local_rank.mpi_rank()];
              }
            }

            // transferring_chunk_displacements[first...last]
            //  first == transferring_chunk_displacements_first_indices[target_local_rank]
            //  last == transferring_chunk_displacements_first_indices[target_local_rank+1]
            auto transferring_chunk_displacements = std::vector<int>(num_transferring_chunks);
            auto transferring_chunk_displacements_first_indices
              = std::vector<std::vector<int>::size_type>(num_local_processes + decltype(num_local_processes){1u});
            std::partial_sum(
              begin(num_transferring_chunks_vec), end(num_transferring_chunks_vec),
              std::next(begin(transferring_chunk_displacements_first_indices)));
            auto transferring_chunk_displacements_indices = transferring_chunk_displacements_first_indices;
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, source_rank_in_unit);
            auto const chunk_size = data_block_size / last_swap_qubit_value;
            for (auto chunk_index = decltype(num_chunks){0}; chunk_index < num_chunks; ++chunk_index)
            {
              auto const target_local_rank = target_local_ranks[chunk_index];
              if (target_local_rank == yampi::null_process)
                continue;

              auto const index = transferring_chunk_displacements_indices[target_local_rank.mpi_rank()]++;
              transferring_chunk_displacements[index] = static_cast<int>(chunk_index * chunk_size);
            }

            auto counts = std::vector<int>(num_local_processes, 1);
            auto const displacements = std::vector<int>(num_local_processes, 0);
            auto datatypes = std::vector<yampi::datatype>(num_local_processes);

            auto const last_target_local_rank = yampi::rank{static_cast<int>(num_local_processes)};
            for (auto target_local_rank = yampi::rank{0}; target_local_rank < last_target_local_rank; ++target_local_rank)
            {
              auto const transferring_chunk_displacements_first = begin(transferring_chunk_displacements);
              auto const first_index = transferring_chunk_displacements_first_indices[target_local_rank.mpi_rank()];
              auto const last_index = transferring_chunk_displacements_first_indices[target_local_rank.mpi_rank() + 1];

              datatypes[target_local_rank.mpi_rank()]
                = yampi::datatype{
                    datatype,
                    yampi::fixed_blocks{
                      yampi::count{chunk_size},
                      transferring_chunk_displacements_first + first_index,
                      transferring_chunk_displacements_first + last_index},
                    environment};
            }

            {
              ::ket::mpi::utility::log_with_time_guard<char> print{
                ::ket::mpi::utility::generate_logger_string(std::string{"interchange_qubits<"}, num_qubits_of_operation, ">::swap"),
                environment};
              noncontiguous_complete_exchange_qubits(local_state, counts, displacements, datatypes, local_communicator, environment);
            }

            if (intraprocess_swap_chunk_index_pairs.empty())
              return;

            ::ket::mpi::utility::log_with_time_guard<char> print{"swap_local_data", environment};
            for (auto const& chunk_index_pair: intraprocess_swap_chunk_index_pairs)
            {
              auto const first = begin(local_state) + chunk_index_pair.first * chunk_size;
              std::swap_ranges(first, first + chunk_size, begin(local_state) + chunk_index_pair.second * chunk_size);
            }
          }
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

          template <typename LocalState, typename Function>
          static auto interchange_qubits_p2p(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState& local_state,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& interchange_qubits)
          -> void
          {
            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);

            using permutated_qubit_type = ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >;
            auto const least_global_permutated_qubit = least_unit_permutated_qubit + mpi_policy.num_unit_qubits();
            using std::begin;
            using std::end;
            auto const num_permutated_unit_swap_qubits
              = std::count_if(
                  begin(permutated_nonlocal_swap_qubits), end(permutated_nonlocal_swap_qubits),
                  [least_global_permutated_qubit](permutated_qubit_type const permutated_qubit)
                  { return permutated_qubit < least_global_permutated_qubit; });

            if (num_permutated_unit_swap_qubits == 0)
            {
              auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);

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
                    local_state, data_block_index, data_block_size,
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
                begin(permutated_nonlocal_qubit_index_pairs), end(permutated_nonlocal_qubit_index_pairs),
                [](permutated_nonlocal_qubit_index_pair_type const& lhs, permutated_nonlocal_qubit_index_pair_type const& rhs)
                { return lhs.first < rhs.first;});

              // initialization of permutated_nonlocal_qubit_masks (000001000000, 000000001000, 001000000000)
              auto permutated_nonlocal_qubit_masks = std::array<StateInteger, num_qubits_of_operation>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                permutated_nonlocal_qubit_masks[index]
                  = (StateInteger{1u} << permutated_nonlocal_swap_qubits[index]) >> least_unit_permutated_qubit;

              // initialization of nonlocal_qubit_value_masks (000000111, 000011000, 001100000, 110000000)
              auto nonlocal_qubit_value_masks
                = std::array<StateInteger, num_qubits_of_operation + std::size_t{1u}>{};
              for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                nonlocal_qubit_value_masks[index]
                  = (permutated_nonlocal_qubit_masks[permutated_nonlocal_qubit_index_pairs[index].second] >> index)
                    - StateInteger{1u};
              nonlocal_qubit_value_masks[num_qubits_of_operation] = compl StateInteger{0u};

              using std::rbegin;
              using std::rend;
              std::transform(
                rbegin(nonlocal_qubit_value_masks), std::prev(rend(nonlocal_qubit_value_masks)),
                std::next(rbegin(nonlocal_qubit_value_masks)), rbegin(nonlocal_qubit_value_masks),
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
                // xx0xx0xx0xxx
                auto nonlocal_qubit_value_base = StateInteger{0u};
                for (auto index = std::size_t{0u}; index <= num_qubits_of_operation; ++index)
                  nonlocal_qubit_value_base
                    |= (nonlocal_qubit_value_wo_qubits bitand nonlocal_qubit_value_masks[index]) << index;

                auto const last_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(num_qubits_of_operation);
                // 000, 001, 010, 011, 100, 101, 110
                for (auto qubit_mask1 = StateInteger{0u};
                     qubit_mask1 < last_qubit_mask - StateInteger{1u}; ++qubit_mask1)
                {
                  // xxbxxb'xxb''xxx
                  auto nonlocal_qubit_value1 = nonlocal_qubit_value_base;
                  for (auto index = BitInteger{0u}; index < num_qubits_of_operation; ++index)
                    nonlocal_qubit_value1
                      |= ((qubit_mask1 bitand (StateInteger{1u} << index)) >> index)
                         << (permutated_nonlocal_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second]
                             - least_unit_permutated_qubit);

                  auto const unit_qubit_value1
                    = nonlocal_qubit_value1
                      bitand (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy) - StateInteger{1u});
                  auto const rank_in_unit1 = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value1);
                  auto const global_qubit_value1
                    = (nonlocal_qubit_value1
                       bitand ((::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment) - StateInteger{1u})
                               << mpi_policy.num_unit_qubits()))
                      >> mpi_policy.num_unit_qubits();
                  auto const rank1 = ::ket::mpi::utility::policy::rank(mpi_policy, global_qubit_value1, rank_in_unit1);

                  // (qubit_mask1 + 1), ..., 111
                  for (auto qubit_mask2 = qubit_mask1 + StateInteger{1u};
                       qubit_mask2 < last_qubit_mask; ++qubit_mask2)
                  {
                    // xxcxxc'xxc''xxx
                    auto nonlocal_qubit_value2 = nonlocal_qubit_value_base;
                    for (auto index = BitInteger{0u}; index < num_qubits_of_operation; ++index)
                      nonlocal_qubit_value2
                        |= ((qubit_mask2 bitand (StateInteger{1u} << index)) >> index)
                           << (permutated_nonlocal_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second]
                               - least_unit_permutated_qubit);

                    auto const unit_qubit_value2
                      = nonlocal_qubit_value2
                        bitand (::ket::mpi::utility::policy::num_unit_qubit_values(mpi_policy) - StateInteger{1u});
                    auto const rank_in_unit2 = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, unit_qubit_value2);
                    auto const global_qubit_value2
                      = (nonlocal_qubit_value2
                         bitand ((::ket::mpi::utility::policy::num_units(mpi_policy, communicator, environment) - StateInteger{1u})
                                 << mpi_policy.num_unit_qubits()))
                        >> mpi_policy.num_unit_qubits();
                    auto const rank2 = ::ket::mpi::utility::policy::rank(mpi_policy, global_qubit_value2, rank_in_unit2);

                    if (rank2 == present_rank)
                    {
                      // (0000000|0000|)bb'b''00000
                      auto local_first_index2 = StateInteger{0u};
                      for (auto index = std::size_t{0u}; index < num_qubits_of_operation; ++index)
                        local_first_index2
                          |= ((qubit_mask1 bitand (StateInteger{1u} << index)) >> index)
                             << permutated_local_swap_qubits[permutated_nonlocal_qubit_index_pairs[index].second];

                      if (rank1 == present_rank)
                      {
                        // (0000000|0000|)cc'c''00000
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
                      // (0000000|0000|)cc'c''00000
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
          }

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <typename NoncontiguousIterator>
          struct interchange_qubits_dispatch2
          {
            template <typename LocalState, typename Function1, typename DerivedDatatype, typename Function2>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&& interchange_qubits, yampi::datatype_base<DerivedDatatype> const&, Function2&&)
            -> void
            {
              interchange_qubits_p2p(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
                communicator, environment, std::forward<Function1>(interchange_qubits));
            }
          }; // struct interchange_qubits_dispatch2<NoncontiguousIterator>

          template <typename T>
          struct interchange_qubits_dispatch2<T*>
          {
            template <typename LocalState, typename Function1, typename DerivedDatatype, typename Function2>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&&, yampi::datatype_base<DerivedDatatype> const& datatype, Function2&& complete_exchange_qubits)
            -> void
            {
              interchange_qubits_collective(
                mpi_policy, local_state, datatype,
                permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
                communicator, environment, std::forward<Function2>(complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch2<T*>

          template <typename LocalState_>
          struct interchange_qubits_dispatch1
          {
            template <typename LocalState, typename Function1, typename DerivedDatatype, typename Function2>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              LocalState& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&& interchange_qubits, yampi::datatype_base<DerivedDatatype> const& datatype, Function2&& noncontiguous_complete_exchange_qubits)
            -> void
            {
              interchange_qubits_dispatch2< ::ket::utility::meta::iterator_t<LocalState_> >::call(
                mpi_policy, local_state,
                permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
                communicator, environment,
                std::forward<Function1>(interchange_qubits),
                datatype, std::forward<Function2>(noncontiguous_complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch1<LocalState_>

          template <typename Complex, typename LocalStateAllocator>
          struct interchange_qubits_dispatch1<std::vector<Complex, LocalStateAllocator>>
          {
            template <typename Function1, typename DerivedDatatype, typename Function2>
            static auto call(
              ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
              std::vector<Complex, LocalStateAllocator>& local_state,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_local_swap_qubits,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
              std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
              yampi::communicator const& communicator, yampi::environment const& environment,
              Function1&&, yampi::datatype_base<DerivedDatatype> const& datatype, Function2&& noncontiguous_complete_exchange_qubits)
            -> void
            {
              interchange_qubits_collective(
                mpi_policy, local_state, datatype,
                permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
                communicator, environment, std::forward<Function2>(noncontiguous_complete_exchange_qubits));
            }
          }; // struct interchange_qubits_dispatch1<std::vector<Complex, LocalStateAllocator>>
# endif // KET_USE_COLLECTIVE_COMMUNICATIONS

# ifdef KET_USE_COLLECTIVE_COMMUNICATIONS
          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename Function1, typename DerivedDatatype, typename Function2>
          static auto do_call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function1&& interchange_qubits, yampi::datatype_base<DerivedDatatype> const& datatype, Function2&& noncontiguous_complete_exchange_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto permutated_local_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            initialize_local_swap_qubits(
              mpi_policy, parallel_policy, local_swap_qubits, permutated_local_swap_qubits,
              local_state, least_unit_permutated_qubit, unswappable_qubits, permutation, communicator, environment);

            interchange_qubits_dispatch1<LocalState>::call(
              mpi_policy, local_state,
              permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              communicator, environment,
              std::forward<Function1>(interchange_qubits), datatype, std::forward<Function2>(noncontiguous_complete_exchange_qubits));

            update_permutation(permutation, qubits, local_swap_qubits, communicator, environment);
          }
# else //KET_USE_COLLECTIVE_COMMUNICATIONS
          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename Function>
          static auto do_call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_qubits_of_operation > const& permutated_nonlocal_swap_qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& interchange_qubits)
          -> void
          {
            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto permutated_local_swap_qubits = std::array<permutated_qubit_type, num_qubits_of_operation>{};
            auto local_swap_qubits = std::array<qubit_type, num_qubits_of_operation>{};
            initialize_local_swap_qubits(
              mpi_policy, parallel_policy, local_swap_qubits, permutated_local_swap_qubits,
              local_state, least_unit_permutated_qubit, unswappable_qubits, permutation, communicator, environment);

            interchange_qubits_p2p(
              mpi_policy, local_state,
              permutated_local_swap_qubits, least_unit_permutated_qubit, permutated_nonlocal_swap_qubits,
              communicator, environment, std::forward<Function>(interchange_qubits));

            update_permutation(permutation, qubits, local_swap_qubits, communicator, environment);
          }
# endif //KET_USE_COLLECTIVE_COMMUNICATIONS

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static auto call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            using std::begin;
            using std::end;
            std::copy(
              begin(qubits) + new_unswappable_qubit_index + 1u, end(qubits),
              std::copy(begin(qubits), begin(qubits) + new_unswappable_qubit_index, begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(begin(unswappable_qubits), end(unswappable_qubits), begin(new_unswappable_qubits));
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
          static auto call_lower_maybe_interchange_qubits(
            std::size_t const new_unswappable_qubit_index,
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation > const& qubits,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const& unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            assert(new_unswappable_qubit_index < num_qubits_of_operation);

            using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
            auto new_qubits = std::array<qubit_type, num_qubits_of_operation - 1u>{};
            using std::begin;
            using std::end;
            std::copy(
              begin(qubits) + new_unswappable_qubit_index + 1u, end(qubits),
              std::copy(begin(qubits), begin(qubits) + new_unswappable_qubit_index, begin(new_qubits)));

            auto new_unswappable_qubits = std::array<qubit_type, num_unswappable_qubits + 1u>{};
            std::copy(begin(unswappable_qubits), end(unswappable_qubits), begin(new_unswappable_qubits));
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
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >&,
            yampi::communicator const&, yampi::environment const&)
          -> void
          { }

          template <
            typename ParallelPolicy, typename LocalState,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const&,
            ParallelPolicy const, LocalState&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, 0u > const&,
            std::array< ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits > const&,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >&,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const&, yampi::environment const&)
          -> void
          { }
        }; // struct maybe_interchange_qubits<0u, ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses>>

        template <typename MpiPolicy>
        struct rank_index_to_qubit_value;

        template <typename StateInteger, typename BitInteger, typename NumProcesses>
        struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
        {
          template <typename LocalState>
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState const& local_state, yampi::rank const rank, StateInteger const index)
          -> StateInteger
          {
            // g
            auto const global_qubit_value = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, rank);
            // r_u
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, rank);
            // 2^L
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit);
            // i_u = i / 2^L
            auto const data_block_index = index / data_block_size;
            // l = i % 2^L
            auto const local_qubit_value = index % data_block_size;
            // u
            auto const unit_qubit_value = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, rank_in_unit);

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
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            LocalState const& local_state, StateInteger const qubit_value,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> std::pair<yampi::rank, StateInteger>
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
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            ControlQubits const... control_qubits)
          -> void
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
              mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation,
              present_rank, present_rank_in_unit,
              least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
              std::forward<Function0>(function0),
              std::forward<Function1>(function1),
              unit_permutated_control_qubits, local_permutated_control_qubits,
              control_qubits...);
          }

          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function00, typename Function01, typename Function10, typename Function11,
            typename... ControlQubits>
          static auto call(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            Function00&& function00, Function01&& function01,
            Function10&& function10, Function11&& function11,
            ControlQubits const... control_qubits)
          -> void
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
              mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation,
              present_rank, present_rank_in_unit,
              least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit1, target_qubit2,
              std::forward<Function00>(function00), std::forward<Function01>(function01),
              std::forward<Function10>(function10), std::forward<Function11>(function11),
              unit_permutated_control_qubits, local_permutated_control_qubits,
              control_qubits...);
          }

         private:
          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits,
            typename... ControlQubits>
          static auto call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits const... control_qubits)
          -> void
          {
            auto const permutated_control_qubit = permutation[control_qubit];

            if (permutated_control_qubit < least_unit_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_local_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_local_control_qubits + 1u>{};
              using std::begin;
              using std::end;
              std::copy(
                begin(local_permutated_control_qubits), end(local_permutated_control_qubits),
                begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation,
                present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0), std::forward<Function1>(function1),
                unit_permutated_control_qubits, new_local_permutated_control_qubits,
                control_qubits...);
            }
            else if (permutated_control_qubit < least_global_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_unit_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_unit_control_qubits + 1u>{};
              using std::begin;
              using std::end;
              std::copy(
                begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                begin(new_unit_permutated_control_qubits));
              new_unit_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation,
                present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                std::forward<Function0>(function0), std::forward<Function1>(function1),
                new_unit_permutated_control_qubits, local_permutated_control_qubits,
                control_qubits...);
            }
            else
            {
              static constexpr auto zero_state_integer = StateInteger{0u};
              static constexpr auto one_state_integer = StateInteger{1u};

              auto const mask = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand mask) != zero_state_integer)
                call_impl(
                  mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation,
                  present_rank, present_rank_in_unit,
                  least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit,
                  std::forward<Function0>(function0), std::forward<Function1>(function1),
                  unit_permutated_control_qubits, local_permutated_control_qubits,
                  control_qubits...);
            }
          }

          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits>
          static auto call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            Function0&& function0, Function1&& function1,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits)
          -> void
          {
            auto const permutated_target_qubit = permutation[target_qubit];

            static constexpr auto zero_state_integer = StateInteger{0u};
            static constexpr auto one_state_integer = StateInteger{1u};

            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, present_rank_in_unit);
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, present_rank_in_unit);

            auto const last_local_qubit_value = one_state_integer << least_unit_permutated_qubit;

            if (permutated_target_qubit < least_unit_permutated_qubit)
            {
              auto const target_mask = one_state_integer << permutated_target_qubit;

              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                auto const unit_qubit_value
                  = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                using std::begin;
                using std::end;
                if (std::any_of(
                      begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                      [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                      {
                        return
                          (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                            == zero_state_integer;
                      }))
                  continue;

                ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                  parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                  [&function0, &function1, target_mask](auto const iter, StateInteger const state_integer)
                  {
                    if ((state_integer bitand target_mask) == zero_state_integer)
                      function0(iter, state_integer);
                    else
                      function1(iter, state_integer);
                  });
              }
            }
            else if (permutated_target_qubit < least_global_permutated_qubit)
            {
              auto const target_mask = one_state_integer << (permutated_target_qubit - least_unit_permutated_qubit);

              for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
              {
                auto const unit_qubit_value
                  = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                using std::begin;
                using std::end;
                if (std::any_of(
                      begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                      [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                      {
                        return
                          (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                            == zero_state_integer;
                      }))
                  continue;

                if ((unit_qubit_value bitand target_mask) == zero_state_integer)
                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function0>(function0));
                else
                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function1>(function1));
              }
            }
            else
            {
              auto const target_mask = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

              if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask) == zero_state_integer)
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  auto const unit_qubit_value
                    = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                  using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                  using std::begin;
                  using std::end;
                  if (std::any_of(
                        begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                        [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                        {
                          return
                            (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                              == zero_state_integer;
                        }))
                    continue;

                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function0>(function0));
                }
              else
                for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
                {
                  auto const unit_qubit_value
                    = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

                  using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
                  using std::begin;
                  using std::end;
                  if (std::any_of(
                        begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                        [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                        {
                          return
                            (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                              == zero_state_integer;
                        }))
                    continue;

                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    std::forward<Function1>(function1));
                }
            }
          }

          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function00, typename Function01, typename Function10, typename Function11,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits,
            typename... ControlQubits>
          static auto call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            Function00&& function00, Function01&& function01,
            Function10&& function10, Function11&& function11,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits const... control_qubits)
          -> void
          {
            auto const permutated_control_qubit = permutation[control_qubit];

            if (permutated_control_qubit < least_unit_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_local_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_local_control_qubits + 1u>{};
              using std::begin;
              using std::end;
              std::copy(
                begin(local_permutated_control_qubits), end(local_permutated_control_qubits),
                begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation, present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit1, target_qubit2,
                std::forward<Function00>(function00), std::forward<Function01>(function01),
                std::forward<Function10>(function10), std::forward<Function11>(function11),
                unit_permutated_control_qubits, new_local_permutated_control_qubits,
                control_qubits...);
            }
            else if (permutated_control_qubit < least_global_permutated_qubit)
            {
              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              auto new_unit_permutated_control_qubits
                = std::array<permutated_control_qubit_type, num_unit_control_qubits + 1u>{};
              using std::begin;
              using std::end;
              std::copy(
                begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                begin(new_unit_permutated_control_qubits));
              new_unit_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation, present_rank, present_rank_in_unit,
                least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit1, target_qubit2,
                std::forward<Function00>(function00), std::forward<Function01>(function01),
                std::forward<Function10>(function10), std::forward<Function11>(function11),
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
                  mpi_policy, parallel_policy, std::forward<LocalState>(local_state), permutation, present_rank, present_rank_in_unit,
                  least_unit_permutated_qubit, least_global_permutated_qubit, target_qubit1, target_qubit2,
                  std::forward<Function00>(function00), std::forward<Function01>(function01),
                  std::forward<Function10>(function10), std::forward<Function11>(function11),
                  unit_permutated_control_qubits, local_permutated_control_qubits,
                  control_qubits...);
            }
          }

          template <
            typename ParallelPolicy, typename LocalState, typename Allocator,
            typename Function00, typename Function01, typename Function10, typename Function11,
            std::size_t num_unit_control_qubits, std::size_t num_local_control_qubits>
          static auto call_impl(
            ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> const& mpi_policy,
            ParallelPolicy const parallel_policy, LocalState&& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const present_rank, yampi::rank const present_rank_in_unit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_unit_permutated_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
            Function00&& function00, Function01&& function01,
            Function10&& function10, Function11&& function11,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_unit_control_qubits > const& unit_permutated_control_qubits,
            std::array< ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >, num_local_control_qubits > const& local_permutated_control_qubits)
          -> void
          {
            auto const permutated_target_qubit1 = permutation[target_qubit1];
            auto const permutated_target_qubit2 = permutation[target_qubit2];

            static constexpr auto zero_state_integer = StateInteger{0u};
            static constexpr auto one_state_integer = StateInteger{1u};

            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, present_rank_in_unit);
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, present_rank_in_unit);

            auto const last_local_qubit_value = one_state_integer << least_unit_permutated_qubit;

            for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
            {
              auto const unit_qubit_value
                = ::ket::mpi::utility::policy::unit_qubit_value(mpi_policy, data_block_index, present_rank_in_unit);

              using permutated_control_qubit_type = ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > >;
              using std::begin;
              using std::end;
              if (std::any_of(
                    begin(unit_permutated_control_qubits), end(unit_permutated_control_qubits),
                    [unit_qubit_value, least_unit_permutated_qubit](permutated_control_qubit_type const& permutated_control_qubit)
                    {
                      return
                        (unit_qubit_value bitand (one_state_integer << (permutated_control_qubit - least_unit_permutated_qubit)))
                          == zero_state_integer;
                    }))
                continue;

              if (permutated_target_qubit1 < least_unit_permutated_qubit)
              {
                auto const target_mask1 = one_state_integer << permutated_target_qubit1;

                if (permutated_target_qubit2 < least_unit_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << permutated_target_qubit2;

                  ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                    parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                    [&function00, &function01, &function10, &function11, target_mask1, target_mask2](
                      auto const iter, StateInteger const state_integer)
                    {
                      if ((state_integer bitand target_mask1) == zero_state_integer)
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function00(iter, state_integer);
                        else
                          function10(iter, state_integer);
                      }
                      else
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function01(iter, state_integer);
                        else
                          function11(iter, state_integer);
                      }
                    });
                }
                else if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_unit_permutated_qubit);

                  if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function00, &function01, target_mask1](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask1) == zero_state_integer)
                          function00(iter, state_integer);
                        else
                          function01(iter, state_integer);
                      });
                  else // if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function10, &function11, target_mask1](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask1) == zero_state_integer)
                          function10(iter, state_integer);
                        else
                          function11(iter, state_integer);
                      });
                }
                else // if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_global_permutated_qubit);

                  if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function00, &function01, target_mask1](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask1) == zero_state_integer)
                          function00(iter, state_integer);
                        else
                          function01(iter, state_integer);
                      });
                  else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function10, &function11, target_mask1](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask1) == zero_state_integer)
                          function10(iter, state_integer);
                        else
                          function11(iter, state_integer);
                      });
                }
              }
              else if (permutated_target_qubit1 < least_global_permutated_qubit)
              {
                auto const target_mask1 = one_state_integer << (permutated_target_qubit1 - least_unit_permutated_qubit);

                if (permutated_target_qubit2 < least_unit_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << permutated_target_qubit2;

                  if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function00, &function10, target_mask2](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function00(iter, state_integer);
                        else
                          function10(iter, state_integer);
                      });
                  else // if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function01, &function11, target_mask2](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function01(iter, state_integer);
                        else
                          function11(iter, state_integer);
                      });
                }
                else if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_unit_permutated_qubit);

                  if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                  {
                    if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function00>(function00));
                    else // if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function10>(function10));
                  }
                  else // if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                  {
                    if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function01>(function01));
                    else // if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function11>(function11));
                  }
                }
                else // if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_global_permutated_qubit);

                  if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                  {
                    if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function00>(function00));
                    else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function10>(function10));
                  }
                  else // if ((unit_qubit_value bitand target_mask1) == zero_state_integer)
                  {
                    if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function01>(function01));
                    else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function11>(function11));
                  }
                }
              }
              else // if (permutated_target_qubit1 < least_global_permutated_qubit)
              {
                auto const target_mask1 = one_state_integer << (permutated_target_qubit1 - least_global_permutated_qubit);

                if (permutated_target_qubit2 < least_unit_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << permutated_target_qubit2;

                  if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function00, &function10, target_mask2](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function00(iter, state_integer);
                        else
                          function10(iter, state_integer);
                      });
                  else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                    ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                      parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                      [&function01, &function11, target_mask2](auto const iter, StateInteger const state_integer)
                      {
                        if ((state_integer bitand target_mask2) == zero_state_integer)
                          function01(iter, state_integer);
                        else
                          function11(iter, state_integer);
                      });
                }
                else if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_unit_permutated_qubit);

                  if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                  {
                    if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function00>(function00));
                    else // if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function10>(function10));
                  }
                  else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                  {
                    if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function01>(function01));
                    else // if ((unit_qubit_value bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function11>(function11));
                  }
                }
                else // if (permutated_target_qubit2 < least_global_permutated_qubit)
                {
                  auto const target_mask2 = one_state_integer << (permutated_target_qubit2 - least_global_permutated_qubit);

                  if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                  {
                    if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function00>(function00));
                    else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function10>(function10));
                  }
                  else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask1) == zero_state_integer)
                  {
                    if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function01>(function01));
                    else // if ((::ket::mpi::utility::policy::global_qubit_value(mpi_policy, present_rank) bitand target_mask2) == zero_state_integer)
                      ::ket::mpi::utility::detail::for_each_in_diagonal_loop(
                        parallel_policy, std::forward<LocalState>(local_state), data_block_index, data_block_size, last_local_qubit_value, local_permutated_control_qubits,
                        std::forward<Function11>(function11));
                  }
                }
              }
            }
          }
        }; // struct diagonal_loop< ::ket::mpi::utility::policy::unit_mpi<StateInteger, BitInteger, NumProcesses> >
# endif // KET_USE_DIAGONAL_LOOP
      } // namespace dispatch
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_UNIT_MPI_HPP
