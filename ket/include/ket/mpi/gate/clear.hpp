#ifndef KET_MPI_GATE_CLEAR_HPP
# define KET_MPI_GATE_CLEAR_HPP

# include <boost/config.hpp>

# include <vector>
# include <array>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/clear.hpp>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   include <ket/mpi/permutated.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/clear.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // CLEAR_i
      // CLEAR_1 (a_{0} |0> + a_{1} |1>) = |0>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace clear_detail
      {
        template <typename ParallelPolicy, typename Qubit>
        struct call_clear
        {
          ParallelPolicy parallel_policy_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_clear(ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          { ::ket::gate::clear(parallel_policy_, first, last, permutated_qubit_.qubit()); }
        }; // struct call_clear<ParallelPolicy, Qubit>

        template <typename ParallelPolicy, typename Qubit>
        inline call_clear<ParallelPolicy, Qubit> make_call_clear(
          ParallelPolicy const parallel_policy, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, permutated_qubit}; }
      } // namespace clear_detail

# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      namespace local
      {
        template <
          typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& clear(
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
            return ::ket::mpi::gate::page::clear(parallel_policy, local_state, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, permutated_qubit](auto const first, auto const last)
            { ::ket::gate::clear(parallel_policy, first, last, permutated_qubit.qubit()); });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::clear_detail::make_call_clear(parallel_policy, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace local

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Clear "}, qubit), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);

        return ::ket::mpi::gate::local::clear(
          mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline RandomAccessRange& clear(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Clear "}, qubit), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, datatype, communicator, environment);

        return ::ket::mpi::gate::local::clear(
          mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Clear "}, qubit), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);

        return ::ket::mpi::gate::local::clear(
          mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline RandomAccessRange& clear(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Clear "}, qubit), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, datatype, communicator, environment);

        return ::ket::mpi::gate::local::clear(
          mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline RandomAccessRange& clear(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& clear(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      // [[deprecated]]
      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline RandomAccessRange& clear(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& clear(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline RandomAccessRange& clear(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      {
        return ::ket::mpi::gate::clear(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_CLEAR_HPP
