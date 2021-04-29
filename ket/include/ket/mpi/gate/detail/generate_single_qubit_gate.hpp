#ifndef KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE_HPP
# define KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE_HPP

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   define KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE(gate_name, gate_symbol) \
namespace ket\
{\
  namespace mpi\
  {\
    namespace gate\
    {\
      namespace gate_name ## _detail\
      {\
        template <\
          typename MpiPolicy, typename ParallelPolicy,\
          typename RandomAccessRange,\
          typename StateInteger, typename BitInteger, typename Allocator>\
        inline RandomAccessRange& gate_name(\
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
          RandomAccessRange& local_state,\
          ::ket::qubit<StateInteger, BitInteger> const qubit,\
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
          yampi::communicator const& communicator, yampi::environment const& environment)\
        {\
          if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
            return ::ket::mpi::gate::page::gate_name(\
              mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
          auto const permutated_qubit = permutation[qubit];\
          return ::ket::mpi::utility::for_each_local_range(\
            mpi_policy, local_state, communicator, environment,\
            [parallel_policy, permutated_qubit](auto const first, auto const last)\
            { ::ket::gate::gate_name(parallel_policy, first, last, permutated_qubit); });\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{#gate_symbol " "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{#gate_symbol " "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, datatype, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
      }\
\
      namespace gate_name ## _detail\
      {\
        template <\
          typename MpiPolicy, typename ParallelPolicy,\
          typename RandomAccessRange,\
          typename StateInteger, typename BitInteger, typename Allocator>\
        inline RandomAccessRange& adj_ ## gate_name(\
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
          RandomAccessRange& local_state,\
          ::ket::qubit<StateInteger, BitInteger> const qubit,\
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
          yampi::communicator const& communicator, yampi::environment const& environment)\
        {\
          if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
            return ::ket::mpi::gate::page::adj_ ## gate_name(\
              mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
          auto const permutated_qubit = permutation[qubit];\
          return ::ket::mpi::utility::for_each_local_range(\
            mpi_policy, local_state, communicator, environment,\
            [parallel_policy, permutated_qubit](auto const first, auto const last)\
            { ::ket::gate::adj_ ## gate_name(parallel_policy, first, last, permutated_qubit); });\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(" #gate_symbol ") "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::adj_ ## gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(" #gate_symbol ") "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, datatype, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::adj_ ## gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
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
            : parallel_policy_{parallel_policy},\
              qubit_{qubit}\
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
          return call_ ## gate_name<ParallelPolicy, Qubit>{\
            parallel_policy, qubit};\
        }\
\
        template <\
          typename MpiPolicy, typename ParallelPolicy,\
          typename RandomAccessRange,\
          typename StateInteger, typename BitInteger, typename Allocator>\
        inline RandomAccessRange& gate_name(\
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
          RandomAccessRange& local_state,\
          ::ket::qubit<StateInteger, BitInteger> const qubit,\
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
          yampi::communicator const& communicator, yampi::environment const& environment)\
        {\
          if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
            return ::ket::mpi::gate::page::gate_name(\
              mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
          return ::ket::mpi::utility::for_each_local_range(\
            mpi_policy, local_state, communicator, environment,\
            ::ket::mpi::gate::gate_name ## _detail::make_call_ ## gate_name(\
              parallel_policy, permutation[qubit]));\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{#gate_symbol " "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{#gate_symbol " "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, datatype, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
      }\
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
            : parallel_policy_{parallel_policy},\
              qubit_{qubit}\
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
          return call_adj_ ## gate_name<ParallelPolicy, Qubit>{\
            parallel_policy, qubit};\
        }\
\
        template <\
          typename MpiPolicy, typename ParallelPolicy,\
          typename RandomAccessRange,\
          typename StateInteger, typename BitInteger, typename Allocator>\
        inline RandomAccessRange& adj_ ## gate_name(\
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
          RandomAccessRange& local_state,\
          ::ket::qubit<StateInteger, BitInteger> const qubit,\
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
          yampi::communicator const& communicator, yampi::environment const& environment)\
        {\
          if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))\
            return ::ket::mpi::gate::page::adj_ ## gate_name(\
              mpi_policy, parallel_policy, local_state, qubit, permutation);\
\
          return ::ket::mpi::utility::for_each_local_range(\
            mpi_policy, local_state, communicator, environment,\
            ::ket::mpi::gate::gate_name ## _detail::make_call_adj_ ## gate_name(\
              parallel_policy, permutation[qubit]));\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(" #gate_symbol ") "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::adj_ ## gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
      }\
\
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(" #gate_symbol ") "}, qubit), environment};\
\
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;\
        auto qubits = std::array<qubit_type, 1u>{qubit};\
        ::ket::mpi::utility::maybe_interchange_qubits(\
          mpi_policy, parallel_policy,\
          local_state, qubits, permutation, buffer, datatype, communicator, environment);\
\
        return ::ket::mpi::gate::gate_name ## _detail::adj_ ## gate_name(\
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(),\
          ::ket::utility::policy::make_sequential(),\
          local_state, qubit, permutation, buffer, datatype, communicator, environment);\
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
        yampi::communicator const& communicator,\
        yampi::environment const& environment)\
      {\
        return ::ket::mpi::gate::adj_ ## gate_name(\
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,\
          local_state, qubit, permutation, buffer, communicator, environment);\
      }\
\
      template <\
        typename ParallelPolicy, typename RandomAccessRange,\
        typename StateInteger, typename BitInteger,\
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>\
      inline RandomAccessRange& adj_ ## gate_name(\
        ParallelPolicy const parallel_policy,\
        RandomAccessRange& local_state,\
        ::ket::qubit<StateInteger, BitInteger> const qubit,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,\
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,\
        yampi::datatype_base<DerivedDatatype> const& datatype,\
        yampi::communicator const& communicator,\
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

#endif // KET_MPI_GATE_DETAIL_GENERATE_SINGLE_QUBIT_GATE_HPP
