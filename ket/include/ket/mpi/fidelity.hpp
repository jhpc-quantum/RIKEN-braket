#ifndef KET_MPI_FIDELITY_HPP
# define KET_MPI_FIDELITY_HPP

# include <complex>
# include <vector>
# include <string>
# include <type_traits>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/inner_product.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    // |<Psi_2|Psi_1>|^2
    // (1) two states |Psi_1> and |Psi_2> exist in the same MPI group
    // all_reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, communicator, environment));
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, datatype, communicator, environment));
    }

    template <typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, datatype, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, datatype, communicator, environment);
    }

    // reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state1, local_state2, root, communicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state1, local_state2, datatype, root, communicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, root, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, datatype, root, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, root, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, datatype, root, communicator, environment);
    }

    // |<Psi_{remote}|Psi_{local}>|^2
    // (2) two states |Psi_{local}> and |Psi_{remote}> do not exist in the same MPI group
    // all_reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, intracommunicator, intercommunicator, environment));
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, datatype, intracommunicator, intercommunicator, environment));
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, intracommunicator, intercommunicator, environment);
    }

    // reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, root, intracommunicator, intercommunicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, root, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, root, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment);
    }

    // <Psi_k|Psi_0> (k = 0, ..., N_{states}: value of intercircuit_communicator.rank(environment))
    // (3) states |Psi_k> do not exist in the same MPI group
    // all_reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment));
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment));
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    // reduce version
    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Fidelity"}), environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    // |<Psi_2| A_{ij} |Psi_1>|^2
    // (1) two states |Psi_1> and |Psi_2> exist in the same MPI group
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState1> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // <Psi_{remote}| A_{ij} |Psi_{local}>
    // (2) two states |Psi_{local}> and |Psi_{remote}> do not exist in the same MPI group
    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // <Psi_k| A_{ij} |Psi_0> (k = 0, ..., N_{states}: value of intercircuit_communicator.rank(environment))
    // (3) states |Psi_k> do not exist in the same MPI group
    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using std::norm;
      return norm(::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...));
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Fidelity with observable:"}, qubit, qubits...),
        environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...)
      .map([](complex_type const& inner_product) { using std::norm; return norm(inner_product); });
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    fidelity(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::fidelity(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_FIDELITY_HPP
