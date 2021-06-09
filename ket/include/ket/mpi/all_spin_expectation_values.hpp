#ifndef KET_MPI_ALL_EXPECTATION_VALUES_HPP
# define KET_MPI_ALL_EXPECTATION_VALUES_HPP

# include <array>
# include <type_traits>

# include <boost/optional.hpp>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/spin_expectation_value.hpp>
# include <ket/mpi/utility/general_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    // all_reduce version
    template <
      typename SpinsAllocator,
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto result = std::vector<spin_type, SpinsAllocator>{};
      result.reserve(num_qubits);

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto const last_qubit = qubit_type{num_qubits};
      for (auto qubit = qubit_type{BitInteger{0u}}; qubit < last_qubit; ++qubit)
        result.push_back(
          ::ket::mpi::spin_expectation_value(
            mpi_policy, parallel_policy, local_state, qubit, permutation, buffer,
            communicator, environment));

      return result;
    }

    template <
      typename SpinsAllocator,
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto result = std::vector<spin_type, SpinsAllocator>{};
      result.reserve(num_qubits);

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto const last_qubit = qubit_type{num_qubits};
      for (auto qubit = qubit_type{BitInteger{0u}}; qubit < last_qubit; ++qubit)
        result.push_back(
          ::ket::mpi::spin_expectation_value(
            mpi_policy, parallel_policy, local_state, qubit, permutation, buffer,
            real_datatype, complex_datatype, communicator, environment));

      return result;
    }

    template <
      typename SpinsAllocator,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }


    /*
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        mpi_policy, parallel_policy,
        local_state, permutation, num_qubits,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        mpi_policy, parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }
    */


    // reduce version
    template <
      typename SpinsAllocator,
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      auto const is_root = communicator.rank(environment) == root;

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto result = std::vector<spin_type, SpinsAllocator>{};
      if (is_root)
        result.reserve(num_qubits);

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto const last_qubit = qubit_type{num_qubits};
      for (auto qubit = qubit_type{BitInteger{0u}}; qubit < last_qubit; ++qubit)
      {
        auto const maybe_expectation_value
          = ::ket::mpi::spin_expectation_value(
              mpi_policy, parallel_policy, local_state, qubit, permutation, buffer,
              root, communicator, environment);

        if (is_root and maybe_expectation_value)
          result.push_back(*maybe_expectation_value);
      }

      if (not is_root)
        return boost::none;

      return result;
    }

    template <
      typename SpinsAllocator,
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      auto const is_root = communicator.rank(environment) == root;

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto result = std::vector<spin_type, SpinsAllocator>{};
      if (is_root)
        result.reserve(num_qubits);

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto const last_qubit = qubit_type{num_qubits};
      for (auto qubit = qubit_type{BitInteger{0u}}; qubit < last_qubit; ++qubit)
      {
        auto const maybe_expectation_value
          = ::ket::mpi::spin_expectation_value(
              mpi_policy, parallel_policy, local_state, qubit, permutation, buffer,
              real_datatype, complex_datatype, root, communicator, environment);

        if (is_root and maybe_expectation_value)
          result.push_back(*maybe_expectation_value);
      }

      if (not is_root)
        return boost::none;

      return result;
    }

    template <
      typename SpinsAllocator,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, root, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, root, communicator, environment);
    }

    template <
      typename SpinsAllocator,
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values<SpinsAllocator>(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }


    /*
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, root, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          std::array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::all_spin_expectation_values(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }
    */
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_ALL_EXPECTATION_VALUES_HPP
