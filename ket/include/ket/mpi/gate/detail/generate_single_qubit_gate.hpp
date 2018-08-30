#ifndef KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE_HPP
# define KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   define KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(gate_name, gate_symbol) \
namespace ket\
{\
  namespace mpi\
  {\
    namespace gate\
    {\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        std::ostringstream output_string_stream(#gate_symbol " ", std::ios_base::ate);\
        output_string_stream << qubit;\
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);\
\
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
        KET_array<qubit_type, 1u> qubits = { qubit };\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation,\
          buffer, datatype, communicator, environment);\
\
        if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
          return ::ket::mpi::gate::page::gate_name(\
            mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
        qubit_type const permutated_qubit = permutation[qubit];\
        return ::ket::mpi::utility::for_each_local_range(\
          mpi_policy, local_state,\
          [parallel_policy, permutated_qubit](auto const first, auto const last)\
          { ::ket::gate::gate_name(parallel_policy, first, last, permutated_qubit); });\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        std::ostringstream output_string_stream("Adj(" #gate_symbol ") ", std::ios_base::ate);\
        output_string_stream << qubit;\
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);\
\
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
        KET_array<qubit_type, 1u> qubits = { qubit };\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation,\
          buffer, datatype, communicator, environment);\
\
        if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
          return ::ket::mpi::gate::page::adj_ ## gate_name(\
            mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
        qubit_type const permutated_qubit = permutation[qubit];\
        return ::ket::mpi::utility::for_each_local_range(\
          mpi_policy, local_state,\
          [parallel_policy, permutated_qubit](auto const first, auto const last)\
          { ::ket::gate::adj_ ## gate_name(parallel_policy, first, last, permutated_qubit); });\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
    }\
  }\
}
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
#   define KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(gate_name, gate_symbol) \
namespace ket\
{\
  namespace mpi\
  {\
    namespace gate\
    {\
      namespace gate_name ## _detail\
      {\
        template <typename ParallelPolicy, typename Qubit>\
        struct call_ ## gate_name\
        {\
          ParallelPolicy parallel_policy_;\
          Qubit qubit_;\
\
          call_ ## gate_name(ParallelPolicy const parallel_policy, Qubit const qubit)\
            : parallel_policy_(parallel_policy),\
              qubit_(qubit)\
          { }\
\
          template <typename RandomAccessIterator>\
          void operator()(\
            RandomAccessIterator const first,\
            RandomAccessIterator const last) const\
          { ::ket::gate::gate_name(parallel_policy_, first, last, qubit_); }\
        };\
\
        template <typename ParallelPolicy, typename Qubit>\
        inline call_ ## gate_name<ParallelPolicy, Qubit> make_call_ ## gate_name(\
          ParallelPolicy const parallel_policy, Qubit const qubit)\
        {\
          return call_ ## gate_name<ParallelPolicy, Qubit>(\
            parallel_policy, qubit);\
        }\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        std::ostringstream output_string_stream(#gate_symbol " ", std::ios_base::ate);\
        output_string_stream << qubit;\
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);\
\
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
        KET_array<qubit_type, 1u> qubits = { qubit };\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation,\
          buffer, datatype, communicator, environment);\
\
        if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
          return ::ket::mpi::gate::page::gate_name(\
            mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
        return ::ket::mpi::utility::for_each_local_range(\
          mpi_policy, local_state,\
          ::ket::mpi::gate::gate_name ## _detail::make_call_ ## gate_name(\
            parallel_policy, permutation[qubit]));\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
\
      namespace gate_name ## _detail\
      {\
        template <typename ParallelPolicy, typename Qubit>\
        struct call_adj_ ## gate_name\
        {\
          ParallelPolicy parallel_policy_;\
          Qubit qubit_;\
\
          call_adj_ ## gate_name(ParallelPolicy const parallel_policy, Qubit const qubit)\
            : parallel_policy_(parallel_policy),\
              qubit_(qubit)\
          { }\
\
          template <typename RandomAccessIterator>\
          void operator()(\
            RandomAccessIterator const first,\
            RandomAccessIterator const last) const\
          { ::ket::gate::adj_ ## gate_name(parallel_policy_, first, last, qubit_); }\
        };\
\
        template <typename ParallelPolicy, typename Qubit>\
        inline call_adj_ ## gate_name<ParallelPolicy, Qubit> make_call_adj_ ## gate_name(\
          ParallelPolicy const parallel_policy, Qubit const qubit)\
        {\
          return call_adj_ ## gate_name<ParallelPolicy, Qubit>(\
            parallel_policy, qubit);\
        }\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        std::ostringstream output_string_stream("Adj(" #gate_symbol ") ", std::ios_base::ate);\
        output_string_stream << qubit;\
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);\
\
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
        KET_array<qubit_type, 1u> qubits = { qubit };\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation,\
          buffer, datatype, communicator, environment);\
\
        if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
          return ::ket::mpi::gate::page::adj_ ## gate_name(\
            mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
        return ::ket::mpi::utility::for_each_local_range(\
          mpi_policy, local_state,\
          ::ket::mpi::gate::gate_name ## _detail::make_call_adj_ ## gate_name(\
            parallel_policy, permutation[qubit]));\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator>\
      inline RandomAccessRange& adj_ ## gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype const datatype,\
        yampi::communicator const communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation,\
          buffer, datatype, communicator, environment);\
      }\
    }\
  }\
}
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

#endif

