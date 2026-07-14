#ifndef KET_MPI_GATE_HADAMARD_HPP
# define KET_MPI_GATE_HADAMARD_HPP

# include <vector>
# include <array>
# include <utility>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/hadamard.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/hadamard.hpp>
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
        // H_i
        // H_1 (a_0 |0> + a_1 |1>) = (a_0 + a_1)/sqrt(2) |0> + (a_0 - a_1)/sqrt(2) |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto hadamard(
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
            return ::ket::mpi::gate::page::hadamard(parallel_policy, local_state, permutated_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::hadamard(parallel_policy, first, last, permutated_qubit.qubit()); });
        }

        // CH_{tc} or C1H_{tc}
        // CH_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + a_{11})/sqrt(2) |10> + (a_{10} - a_{11})/sqrt(2) |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto hadamard(
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
              return ::ket::mpi::gate::page::chadamard_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::chadamard_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::chadamard_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::hadamard(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
        }

        // C...CH_{tc...c'} or CnH_{tc...c'}
        namespace dispatch
        {
          template <typename LocalState>
          struct transpage_hadamard
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
          }; // struct transpage_hadamard<LocalState>
        } // namespace dispatch

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename... ControlQubits>
        inline auto hadamard(
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
            ::ket::mpi::gate::local::dispatch::transpage_hadamard<local_state_type>::call(
              parallel_policy, local_state,
              permutation[target_qubit], permutation[control_qubit1],
              permutation[control_qubit2], permutation[control_qubits]...);
          }

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &permutation, target_qubit, control_qubit1, control_qubit2, control_qubits...](auto const first, auto const last)
            {
              ::ket::gate::hadamard(
                parallel_policy, first, last,
                permutation[target_qubit].qubit(), permutation[control_qubit1].qubit(),
                permutation[control_qubit2].qubit(), permutation[control_qubits].qubit()...);
            });
        }
      } // namespace local

      namespace hadamard_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto hadamard(
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

          return ::ket::mpi::gate::local::hadamard(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto hadamard(
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

          return ::ket::mpi::gate::local::hadamard(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace hadamard_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"H "}, qubit), environment};

        return ::ket::mpi::gate::hadamard_detail::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"H "}, qubit), environment};

        return ::ket::mpi::gate::hadamard_detail::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("H"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::hadamard_detail::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto hadamard(
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
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("H"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::hadamard_detail::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto hadamard(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto hadamard(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto hadamard(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto hadamard(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      namespace hadamard_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::hadamard_detail::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::hadamard_detail::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace hadamard_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(H) "}, qubit), environment};

        return ::ket::mpi::gate::hadamard_detail::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(H) "}, qubit), environment};

        return ::ket::mpi::gate::hadamard_detail::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_hadamard(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("H)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::hadamard_detail::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_hadamard(
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
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("H)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::hadamard_detail::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_hadamard(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_hadamard(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_hadamard(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_hadamard(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_hadamard(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_hadamard(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      namespace runtime
      {
        namespace local
        {
          namespace dispatch
          {
            template <typename LocalState>
            struct transpage_hadamard
            {
              template <
                typename ParallelPolicy,
                typename RandomAccessRange, typename StateInteger, typename BitInteger,
                typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_hadamard<LocalState>
          } // namespace dispatch

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename ControlQubitsRange>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            auto const permutated_target_qubit = permutation[target_qubit];
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });
            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              permutated_target_qubit, permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(local_state, permutated_target_qubit, permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_hadamard<local_state_type>::call(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                using permutated_control_qubit_type = ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)>;
                ::ket::gate::runtime::qubit_ranges::hadamard(
                  parallel_policy, first, last, permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](permutated_control_qubit_type const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }
        } // namespace local

        namespace hadamard_detail
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, communicator, environment,
              target_qubit, control_qubits);
          }
        } // namespace hadamard_detail

        namespace ranges
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(std::string(num_control_qubits, 'C').append("H ")),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::hadamard_detail::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(std::string(num_control_qubits, 'C').append("H ")),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::hadamard_detail::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          inline auto hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              target_qubit, control_qubits);
          }
        } // namespace ranges

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename ControlQubitIterator>
        inline auto hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitIterator>
        inline auto hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit);
        }

        namespace ranges
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto adj_hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("H) ")),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::hadamard_detail::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto adj_hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("H) ")),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::hadamard_detail::hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator>
          inline auto adj_hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          inline auto adj_hadamard(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              target_qubit, control_qubits);
          }
        } // namespace ranges

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename ControlQubitIterator>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitIterator>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto adj_hadamard(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit);
        }

        namespace ranges
        {
          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto hadamard(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::hadamard(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto hadamard(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::hadamard(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_hadamard(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_hadamard(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_hadamard(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }
        } // namespace ranges

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto hadamard(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::hadamard(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto hadamard(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::hadamard(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_hadamard(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_hadamard(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_hadamard(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_hadamard(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }
      } // namespace runtime
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_HADAMARD_HPP
