#ifndef KET_MPI_GATE_X_ROTATION_HALF_PI_HPP
# define KET_MPI_GATE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <vector>
# include <array>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/x_rotation_half_pi.hpp>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   include <ket/mpi/permutated.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/x_rotation_half_pi.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace x_rotation_half_pi_detail
      {
        template <typename ParallelPolicy, typename Qubit>
        struct call_x_rotation_half_pi
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_x_rotation_half_pi(ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy}, permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::x_rotation_half_pi(parallel_policy_, first, last, permutated_qubit_.qubit()); }
        }; // struct call_x_rotation_half_pi<ParallelPolicy, Qubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_cx_rotation_half_pi
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<TargetQubit> permutated_target_qubit_;
          ::ket::mpi::permutated<ControlQubit> permutated_control_qubit_;

          call_cx_rotation_half_pi(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
            ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
            : parallel_policy_{parallel_policy},
              permutated_target_qubit_{permutated_target_qubit},
              permutated_control_qubit_{permutated_control_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::x_rotation_half_pi(parallel_policy_, first, last, permutated_target_qubit_.qubit(), permutated_control_qubit_.qubit()); }
        }; // struct call_cx_rotation_half_pi<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename Qubit>
        inline call_x_rotation_half_pi<ParallelPolicy, Qubit> make_call_x_rotation_half_pi(
          ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, permutated_qubit}; }

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_cx_rotation_half_pi<ParallelPolicy, TargetQubit, ControlQubit> make_call_x_rotation_half_pi(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
          ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
        { return {parallel_policy, permutated_target_qubit, permutated_control_qubit}; }
      } // namespace x_rotation_half_pi_detail

# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace local
      {
        // +X_i
        // +X_1 (a_0 |0> + a_1 |1>) = (a_0 + i a_1)/sqrt(2) |0> + (i a_0 + a_1)/sqrt(2) |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        {
          auto const permutated_qubit = permutation[qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit);

          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::x_rotation_half_pi(parallel_policy, local_state, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::x_rotation_half_pi(parallel_policy, first, last, permutated_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::x_rotation_half_pi_detail::make_call_x_rotation_half_pi(parallel_policy, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // C+X_{tc} or C1+X_{tc}
        // C+X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + i a_{11})/sqrt(2) |10> + (i a_{10} + a_{11})/sqrt(2) |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_target_qubit, permutated_control_qubit);

          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::cx_rotation_half_pi_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::cx_rotation_half_pi_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::cx_rotation_half_pi_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::x_rotation_half_pi(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::x_rotation_half_pi_detail::make_call_x_rotation_half_pi(parallel_policy, permutated_target_qubit, permutated_control_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // C...C+X_{tc...c'} or Cn+X_{tc...c'}
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename... ControlQubits>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ControlQubits const... control_qubits)
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          auto const first = std::begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::x_rotation_half_pi(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[target_qubit].qubit(), permutation[control_qubit1].qubit(), permutation[control_qubit2].qubit(), permutation[control_qubits].qubit()...);

          return local_state;
        }
      } // namespace local

      namespace x_rotation_half_pi_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(ControlQubits) + 1u>{target_qubit, ::ket::remove_control(control_qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::local::x_rotation_half_pi(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(ControlQubits) + 1u>{target_qubit, ::ket::remove_control(control_qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::local::x_rotation_half_pi(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace x_rotation_half_pi_detail

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Xpi "}, qubit), environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Xpi "}, qubit), environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("Xpi"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(sizeof...(ControlQubits), 'C').append("Xpi"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace x_rotation_half_pi_detail
      {
        template <typename ParallelPolicy, typename Qubit>
        struct call_adj_x_rotation_half_pi
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_adj_x_rotation_half_pi(ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy}, permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::adj_x_rotation_half_pi(parallel_policy_, first, last, permutated_qubit_.qubit()); }
        }; // struct call_adj_x_rotation_half_pi<ParallelPolicy, Qubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_adj_cx_rotation_half_pi
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<TargetQubit> permutated_target_qubit_;
          ::ket::mpi::permutated<ControlQubit> permutated_control_qubit_;

          call_adj_cx_rotation_half_pi(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
            ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
            : parallel_policy_{parallel_policy},
              permutated_target_qubit_{permutated_target_qubit},
              permutated_control_qubit_{permutated_control_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::adj_x_rotation_half_pi(parallel_policy_, first, last, permutated_target_qubit_.qubit(), permutated_control_qubit_.qubit()); }
        }; // struct call_adj_cx_rotation_half_pi<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename Qubit>
        inline call_adj_x_rotation_half_pi<ParallelPolicy, Qubit> make_call_adj_x_rotation_half_pi(
          ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, permutated_qubit}; }

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_adj_cx_rotation_half_pi<ParallelPolicy, TargetQubit, ControlQubit> make_call_adj_x_rotation_half_pi(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
          ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
        { return {parallel_policy, permutated_target_qubit, permutated_control_qubit}; }
      } // namespace x_rotation_half_pi_detail

# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace local
      {
        // -X_i
        // -X_1 (a_0 |0> + a_1 |1>) = (a_0 - i a_1)/sqrt(2) |0> + (-i a_0 + a_1)/sqrt(2) |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        {
          auto const permutated_qubit = permutation[qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit);

          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::adj_x_rotation_half_pi(parallel_policy, local_state, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::adj_x_rotation_half_pi(parallel_policy, first, last, permutated_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::x_rotation_half_pi_detail::make_call_adj_x_rotation_half_pi(parallel_policy, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // C-X_{tc} or C1-X_{tc}
        // C-X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - i a_{11})/sqrt(2) |10> + (-i a_{10} + a_{11})/sqrt(2) |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_target_qubit, permutated_control_qubit);

          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::adj_cx_rotation_half_pi_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::adj_cx_rotation_half_pi_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::adj_cx_rotation_half_pi_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::adj_x_rotation_half_pi(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::x_rotation_half_pi_detail::make_call_adj_x_rotation_half_pi(parallel_policy, permutated_target_qubit, permutated_control_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // C...C-X_{tc...c'} or Cn-X_{tc...c'}
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename... ControlQubits>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ControlQubits const... control_qubits)
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[target_qubit], permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          auto const first = std::begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::adj_x_rotation_half_pi(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[target_qubit].qubit(), permutation[control_qubit1].qubit(), permutation[control_qubit2].qubit(), permutation[control_qubits].qubit()...);

          return local_state;
        }
      } // namespace local

      namespace x_rotation_half_pi_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... ControlQubits>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(ControlQubits) + 1u>{target_qubit, ::ket::remove_control(control_qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::local::adj_x_rotation_half_pi(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(ControlQubits) + 1u>{target_qubit, ::ket::remove_control(control_qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::local::adj_x_rotation_half_pi(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubit, control_qubits...);
        }
      } // namespace x_rotation_half_pi_detail

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Xpi) "}, qubit), environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Xpi) "}, qubit), environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("Xpi)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(sizeof...(ControlQubits), 'C').append("Xpi)"), target_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::x_rotation_half_pi_detail::adj_x_rotation_half_pi(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, target_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... ControlQubits>
      inline RandomAccessRange& adj_x_rotation_half_pi(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_x_rotation_half_pi(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_X_ROTATION_HALF_PI_HPP
