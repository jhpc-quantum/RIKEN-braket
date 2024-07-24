#ifndef KET_MPI_GATE_PAULI_X_HPP
# define KET_MPI_GATE_PAULI_X_HPP

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
# include <ket/gate/pauli_x.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   include <ket/mpi/permutated.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/pauli_x.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace pauli_x_detail
      {
        template <typename ParallelPolicy, typename Qubit>
        struct call_pauli_x1
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_pauli_x1(ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::pauli_x(parallel_policy_, first, last, permutated_qubit_.qubit()); }
        }; // struct call_pauli_x1<ParallelPolicy, Qubit>

        template <typename ParallelPolicy, typename Qubit>
        struct call_pauli_x2
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<Qubit> permutated_qubit1_;
          ::ket::mpi::permutated<Qubit> permutated_qubit2_;

          call_pauli_x2(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::permutated<Qubit> const permutated_qubit1,
            ::ket::mpi::permutated<Qubit> const permutated_qubit2)
            : parallel_policy_{parallel_policy},
              permutated_qubit1_{permutated_qubit1},
              permutated_qubit2_{permutated_qubit2}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::pauli_x(parallel_policy_, first, last, permutated_qubit1_.qubit(), permutated_qubit2_.qubit()); }
        }; // struct call_pauli_x2<ParallelPolicy, Qubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_pauli_cx
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<TargetQubit> permutated_target_qubit_;
          ::ket::mpi::permutated<ControlQubit> permutated_control_qubit_;

          call_pauli_cx(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
            ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
            : parallel_policy_{parallel_policy},
              permutated_target_qubit_{permutated_target_qubit},
              permutated_control_qubit_{permutated_control_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(RandomAccessIterator const first, RandomAccessIterator const last) const
          { ::ket::gate::pauli_x(parallel_policy_, first, last, permutated_target_qubit_.qubit(), permutated_control_qubit_.qubit()); }
        }; // struct call_pauli_cx<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename Qubit>
        inline call_pauli_x1<ParallelPolicy, Qubit> make_call_pauli_x(
          ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, permutated_qubit}; }

        template <typename ParallelPolicy, typename Qubit>
        inline call_pauli_x2<ParallelPolicy, Qubit> make_call_pauli_x(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::permutated<Qubit> const permutated_qubit1,
          ::ket::mpi::permutated<Qubit> const permutated_qubit2)
        { return {parallel_policy, permutated_qubit1, permutated_qubit2}; }

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_pauli_cx<ParallelPolicy, TargetQubit, ControlQubit> make_call_pauli_x(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
          ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
        { return {parallel_policy, permutated_target_qubit, permutated_control_qubit}; }
      } // namespace pauli_x_detail

# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace local
      {
        // X_i, X1_i, or NOT_i
        // X_1 (a_0 |0> + a_1 |1>) = a_1 |0> + a_0 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_x(
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
            return ::ket::mpi::gate::page::pauli_x1(parallel_policy, local_state, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::pauli_x(parallel_policy, first, last, permutated_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::pauli_x_detail::make_call_pauli_x(parallel_policy, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // XX_{ij} = X_i X_j or X2_{ij}
        // XX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{11} |00> + a_{10} |01> + a_{01} |10> + a_{00} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2)
        {
          auto const permutated_qubit1 = permutation[qubit1];
          auto const permutated_qubit2 = permutation[qubit2];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_qubit1, permutated_qubit2);

          if (::ket::mpi::page::is_on_page(permutated_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
              return ::ket::mpi::gate::page::pauli_x2_2p(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);

            return ::ket::mpi::gate::page::pauli_x2_p(
              parallel_policy, local_state, permutated_qubit1, permutated_qubit2);
          }
          else if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
            return ::ket::mpi::gate::page::pauli_x2_p(
              parallel_policy, local_state, permutated_qubit2, permutated_qubit1);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit1, permutated_qubit2](auto const first, auto const last)
            { ::ket::gate::pauli_x(parallel_policy, first, last, permutated_qubit1.qubit(), permutated_qubit2.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::pauli_x_detail::make_call_pauli_x(parallel_policy, permutated_qubit1, permutated_qubit2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // CX_{tc}, CX1_{tc}, C1X_{tc}, C1X1_{tc}, or CNOT_{tc}
        // CX_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{11} |10> + a_{10} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_x(
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
              return ::ket::mpi::gate::page::pauli_cx_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::pauli_cx_tp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::pauli_cx_cp(
              parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_target_qubit, permutated_control_qubit](auto const first, auto const last)
            { ::ket::gate::pauli_x(parallel_policy, first, last, permutated_target_qubit.qubit(), permutated_control_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::pauli_x_detail::make_call_pauli_x(parallel_policy, permutated_target_qubit, permutated_control_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // C...CX...X_{t...t'c...c'} = C...C(X_t ... X_t')_{c...c'}, CnX...X_{...}, C...CXm_{...}, or CnXm_{...}
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename Qubit2, typename Qubit3, typename... Qubits>
        inline RandomAccessRange& pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[qubit1], permutation[qubit2], permutation[qubit3], permutation[qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          auto const first = std::begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::pauli_x(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[qubit1].qubit(), permutation[qubit2].qubit(), permutation[qubit3].qubit(), permutation[qubits].qubit()...);

          return local_state;
        }
      } // namespace local

      namespace pauli_x_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... Qubits>
        inline RandomAccessRange& pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(Qubits) + 1u>{qubit, ::ket::remove_control(qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::local::pauli_x(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit, qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
        inline RandomAccessRange& pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(Qubits) + 1u>{qubit, ::ket::remove_control(qubits)...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::local::pauli_x(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit, qubits...);
        }
      } // namespace pauli_x_detail

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"X "}, qubit), environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"X "}, qubit), environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"XX "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"XX "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        static constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append(sizeof...(Qubits) + 1u - num_control_qubits, 'X'), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        static constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append(sizeof...(Qubits) + 1u - num_control_qubits, 'X'), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::pauli_x_detail::pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit1, qubit2, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit1, qubit2, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit1, qubit2, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit1, qubit2, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      namespace pauli_x_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... Qubits>
        inline RandomAccessRange& adj_pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        {
          return ::ket::mpi::gate::pauli_x_detail::pauli_x(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, qubit, qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
        inline RandomAccessRange& adj_pauli_x(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        {
          return ::ket::mpi::gate::pauli_x_detail::pauli_x(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
        }
      } // namespace pauli_x_detail

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(X) "}, qubit), environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(X) "}, qubit), environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(XX) "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit1, qubit2);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(XX) "}, qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        static constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append(sizeof...(Qubits) + 1u - num_control_qubits, 'X').append(")"), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        static constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append(sizeof...(Qubits) + 1u - num_control_qubits, 'X').append(")"), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::pauli_x_detail::adj_pauli_x(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit1, qubit2, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit1, qubit2, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit1, qubit2, permutation, buffer, communicator, environment);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit1, qubit2, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline RandomAccessRange& adj_pauli_x(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        return ::ket::mpi::gate::adj_pauli_x(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_PAULI_X_HPP
