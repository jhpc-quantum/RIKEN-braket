#ifndef KET_MPI_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_SPIN_EXPECTATION_VALUE_HPP

# include <array>
# include <ios>
# include <iterator>
# include <type_traits>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/reduce.hpp>
# include <yampi/binary_operation.hpp>

# include <ket/spin_expectation_value.hpp>
# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/page/spin_expectation_value.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, communicator, environment);

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      auto const permutated_qubit = permutation[qubit];
      if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
        spin = ::ket::mpi::page::spin_expectation_value(parallel_policy, local_state, permutated_qubit);
      else
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, permutated_qubit, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutated_qubit.qubit());
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });

      auto result = spin_type{};
      using std::begin;
      yampi::all_reduce(
        yampi::range_to_buffer(spin), begin(result), yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubit);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, complex_datatype, communicator, environment);

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      auto const permutated_qubit = permutation[qubit];
      if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
        spin = ::ket::mpi::page::spin_expectation_value(parallel_policy, local_state, permutated_qubit);
      else
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, permutated_qubit, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutated_qubit.qubit());
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });

      auto result = spin_type{};
      using std::begin;
      yampi::all_reduce(
        yampi::range_to_buffer(spin, real_datatype),
        begin(result), yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer,
        real_datatype, complex_datatype, communicator, environment,
        qubit);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation, buffer, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment, qubit);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, real_datatype,
        complex_datatype, communicator, environment, qubit);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubit, permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment, qubit);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u > >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer,
        real_datatype, complex_datatype, communicator, environment, qubit);
    }


    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, communicator, environment);

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      auto const permutated_qubit = permutation[qubit];
      if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
        spin = ::ket::mpi::page::spin_expectation_value(parallel_policy, local_state, permutated_qubit);
      else
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, permutated_qubit, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutated_qubit.qubit());
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });

      auto result = spin_type{};
      using std::begin;
      yampi::reduce(
        yampi::range_to_buffer(spin), begin(result), yampi::binary_operation(yampi::plus_t()),
        root, communicator, environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, root, communicator, environment,
        qubit);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, complex_datatype, communicator, environment);

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      auto const permutated_qubit = permutation[qubit];
      if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
        spin = ::ket::mpi::page::spin_expectation_value(parallel_policy, local_state, permutated_qubit);
      else
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, permutated_qubit, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutated_qubit.qubit());
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });

      auto result = spin_type{};
      using std::begin;
      yampi::reduce(
        yampi::range_to_buffer(spin, real_datatype), begin(result), yampi::binary_operation(yampi::plus_t()),
        root, communicator, environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, real_datatype, complex_datatype, root, communicator, environment,
        qubit);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation, buffer, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, root, communicator, environment, qubit);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, real_datatype, complex_datatype,
        root, communicator, environment, qubit);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubit, permutation, buffer, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >>>
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, root, communicator, environment, qubit);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<std::array< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >, 3u >> >
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      ::ket::qubit<StateInteger, BitInteger> const qubit)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, real_datatype, complex_datatype,
        root, communicator, environment, qubit);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_SPIN_EXPECTATION_VALUE_HPP
