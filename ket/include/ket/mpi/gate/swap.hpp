#ifndef KET_MPI_GATE_SWAP_HPP
# define KET_MPI_GATE_SWAP_HPP

# include <vector>
# include <array>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/swap.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/swap.hpp>
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
        // SWAP_{ij}
        // SWAP_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{10} |01> + a_{01} |10> + a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2)
        -> RandomAccessRange&
        {
          auto const permutated_qubit1 = permutation[qubit1];
          auto const permutated_qubit2 = permutation[qubit2];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit1, permutated_qubit2);

          if (::ket::mpi::page::is_on_page(permutated_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
              return ::ket::mpi::gate::page::swap_2p(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);

            return ::ket::mpi::gate::page::swap_p(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
          else if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
            return ::ket::mpi::gate::page::swap_p(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1);

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit1, permutated_qubit2](
              auto const first, auto const last)
            {
              ::ket::gate::swap(
                parallel_policy, first, last,
                permutated_qubit1.qubit(), permutated_qubit2.qubit());
            });
        }

        // C...CSWAP_{tt'c...c'} or CnSWAP_{tt'c...c'}
        namespace dispatch
        {
          template <typename LocalState>
          struct transpage_swap
          {
            template <
              typename ParallelPolicy, typename RandomAccessRange,
              typename StateInteger, typename BitInteger, typename... ControlQubits>
            [[noreturn]] static auto call(
              ParallelPolicy const parallel_policy,
              RandomAccessRange& local_state,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit1,
              ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit2,
              ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit,
              ::ket::mpi::permutated<ControlQubits> const... permutated_control_qubits)
            -> RandomAccessRange&
            { throw 1; }
          }; // struct transpage_swap<LocalState>
        } // namespace dispatch

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator,
          typename... ControlQubits>
        inline auto swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[target_qubit1], permutation[target_qubit2], permutation[control_qubit], permutation[control_qubits]...);

          if (::ket::mpi::page::any_on_page(local_state, permutation[target_qubit1], permutation[target_qubit2], permutation[control_qubit], permutation[control_qubits]...))
          {
            using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
            ::ket::mpi::gate::local::dispatch::transpage_swap<local_state_type>::call(
              parallel_policy, local_state,
              permutation[target_qubit1], permutation[target_qubit2], permutation[control_qubit], permutation[control_qubits]...);
          }

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &permutation, target_qubit1, target_qubit2, control_qubit, control_qubits...](auto const first, auto const last)
            {
              ::ket::gate::swap(
                parallel_policy, first, last,
                permutation[target_qubit1].qubit(), permutation[target_qubit2].qubit(),
                permutation[control_qubit].qubit(), permutation[control_qubits].qubit()...);
            });
        }
      } // namespace local

      namespace swap_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);

          return ::ket::mpi::gate::local::swap(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);

          return ::ket::mpi::gate::local::swap(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
        }
      } // namespace swap_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"SWAP "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::swap_detail::swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"SWAP "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::swap_detail::swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("SWAP"), target_qubit1, target_qubit2, control_qubits...),
          environment};

        return ::ket::mpi::gate::swap_detail::swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("SWAP"), target_qubit1, target_qubit2, control_qubits...),
          environment};

        return ::ket::mpi::gate::swap_detail::swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto swap(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto swap(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto swap(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto swap(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      namespace swap_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline auto adj_swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::swap_detail::swap(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline auto adj_swap(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
          ControlQubits const... control_qubits)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::swap_detail::swap(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
        }
      } // namespace swap_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(SWAP) "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::swap_detail::adj_swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(SWAP) "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::swap_detail::adj_swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("SWAP)"), target_qubit1, target_qubit2, control_qubits...),
          environment};

        return ::ket::mpi::gate::swap_detail::adj_swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_swap(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("SWAP)"), target_qubit1, target_qubit2, control_qubits...),
          environment};

        return ::ket::mpi::gate::swap_detail::adj_swap(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_swap(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_swap(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1, ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_swap(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_swap(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline auto adj_swap(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit1,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit2,
        ControlQubits const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_swap(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit1, target_qubit2, control_qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_SWAP_HPP
