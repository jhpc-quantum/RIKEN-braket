#ifndef KET_MPI_GATE_SQRT_PAULI_Y_HPP
# define KET_MPI_GATE_SQRT_PAULI_Y_HPP

# include <vector>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/sqrt_pauli_y.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/sqrt_pauli_y.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/page/any_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace local
      {
        // sY_i
        // sY_1 (a_0 |0> + a_1 |1>) = [(1+i) a_0 - (1+i) a_1]/2 |0> + [(1+i) a_0 + (1+i) a_1]/2 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          auto const permutated_qubit = permutation[qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit);

          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_y(parallel_policy, local_state, permutated_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::sqrt_pauli_y(parallel_policy, first, last, permutated_qubit.qubit()); });
        }

        // CsY_{tc} or C1sY_{tc}
        // CsY_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1+i) a_{10} - (1+i) a_{11}]/2 |10> + [(1+i) a_{10} + (1+i) a_{11}]/2 |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_target_qubit, permutated_control_qubit);

          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::sqrt_pauli_cy_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::sqrt_pauli_cy_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_cy_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::sqrt_pauli_y(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
        }

        // C...CsY_{tc...c'} or CnsY_{tc...c'}
        namespace dispatch
        {
          template <typename LocalState>
          struct transpage_sqrt_pauli_y
          {
            template <
              typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename BitInteger,
              typename... ControlQubits>
            [[noreturn]] static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2,
              ::ket::mpi::permutated<ControlQubits> const... permutated_control_qubits)
            -> RandomAccessRange&
            { throw 1; }
          }; // struct transpage_sqrt_pauli_y<LocalState>
        } // namespace dispatch

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename... ControlQubits>
        inline auto sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...);

          if (::ket::mpi::page::any_on_page(local_state, permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...))
          {
            using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
            ::ket::mpi::gate::local::dispatch::transpage_sqrt_pauli_y<local_state_type>::call(
              parallel_policy, local_state,
              permutation[target_qubit], permutation[control_qubit1],
              permutation[control_qubit2], permutation[control_qubits]...);
          }

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &permutation, target_qubit, control_qubit1, control_qubit2, control_qubits...](auto const first, auto const last)
            {
              ::ket::gate::sqrt_pauli_y(
                parallel_policy, first, last,
                permutation[target_qubit].qubit(), permutation[control_qubit1].qubit(),
                permutation[control_qubit2].qubit(), permutation[control_qubits].qubit()...);
            });
        }
      } // namespace local

      namespace sqrt_pauli_y_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_y(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_y(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace sqrt_pauli_y_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("sY"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_y_detail::sqrt_pauli_y(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("sY"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_y_detail::sqrt_pauli_y(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      namespace local
      {
        // sY+_i
        // sY+_1 (a_0 |0> + a_1 |1>) = [(1-i) a_0 + (1-i) a_1]/2 |0> + [-(1-i) a_0 + (1-i) a_1]/2 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          auto const permutated_qubit = permutation[qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit);

          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_y(parallel_policy, local_state, permutated_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::adj_sqrt_pauli_y(parallel_policy, first, last, permutated_qubit.qubit()); });
        }

        // CsY+_{tc} or C1sY+_{tc}
        // CsY+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + [(1-i) a_{10} + (1-i) a_{11}]/2 |10> + [-(1-i) a_{10} + (1-i) a_{11}]/2 |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_target_qubit, permutated_control_qubit);

          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::adj_sqrt_pauli_cy_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::adj_sqrt_pauli_cy_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_cy_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::adj_sqrt_pauli_y(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
        }

        // C...CsY+_{tc...c'} or CnsY+_{tc...c'}
        namespace dispatch
        {
          template <typename LocalState>
          struct transpage_adj_sqrt_pauli_y
          {
            template <
              typename ParallelPolicy,
              typename RandomAccessRange, typename StateInteger, typename BitInteger,
              typename... ControlQubits>
            [[noreturn]] static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit1,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit2,
              ::ket::mpi::permutated<ControlQubits> const... permutated_control_qubits)
            -> RandomAccessRange&
            { throw 1; }
          }; // struct transpage_adj_sqrt_pauli_y<LocalState>
        } // namespace dispatch

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename... ControlQubits>
        inline auto adj_sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...);

          if (::ket::mpi::page::any_on_page(local_state, permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...))
          {
            using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
            ::ket::mpi::gate::local::dispatch::transpage_adj_sqrt_pauli_y<local_state_type>::call(
              parallel_policy, local_state,
              permutation[target_qubit], permutation[control_qubit1],
              permutation[control_qubit2], permutation[control_qubits]...);
          }

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &permutation, target_qubit, control_qubit1, control_qubit2, control_qubits...](auto const first, auto const last)
            {
              ::ket::gate::adj_sqrt_pauli_y(
                parallel_policy, first, last,
                permutation[target_qubit].qubit(), permutation[control_qubit1].qubit(),
                permutation[control_qubit2].qubit(), permutation[control_qubits].qubit()...);
            });
        }
      } // namespace local

      namespace sqrt_pauli_y_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto adj_sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_y(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto adj_sqrt_pauli_y(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_y(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace sqrt_pauli_y_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("sY)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_y_detail::adj_sqrt_pauli_y(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("sY)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_y_detail::adj_sqrt_pauli_y(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_sqrt_pauli_y(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_y(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_SQRT_PAULI_Y_HPP
