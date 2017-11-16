#ifndef KET_MPI_ALL_EXPECTATION_VALUES_HPP
# define KET_MPI_ALL_EXPECTATION_VALUES_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <boost/optional.hpp>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/spin_expectation_value.hpp>
# include <ket/mpi/utility/general_mpi.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
# else
#   define KET_enable_if boost::enable_if_c
# endif


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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      std::vector<spin_type, SpinsAllocator> result;
      result.reserve(num_qubits);

      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      qubit_type const last_qubit(num_qubits);
      for (qubit_type qubit = static_cast<qubit_type>(static_cast<BitInteger>(0u));
           qubit < last_qubit; ++qubit)
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
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        mpi_policy, parallel_policy,
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::vector<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      bool is_root = communicator.rank(environment) == root;

      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      std::vector<spin_type, SpinsAllocator> result;
      if (is_root)
        result.reserve(num_qubits);

      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      qubit_type const last_qubit(num_qubits);
      for (qubit_type qubit = static_cast<qubit_type>(static_cast<BitInteger>(0u));
           qubit < last_qubit; ++qubit)
      {
        boost::optional<spin_type> const maybe_expectation_value
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
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u>, SpinsAllocator> > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      return ::ket::mpi::all_spin_expectation_values< std::allocator<spin_type> >(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, num_qubits,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::vector<
          KET_array<
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<LocalState>::type>::type, 3u> > > >::type
    all_spin_expectation_values(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      BitInteger const num_qubits,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const communicator,
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


# undef KET_enable_if
# undef KET_array

#endif
