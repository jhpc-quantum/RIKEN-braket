#ifndef KET_MPI_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_SPIN_EXPECTATION_VALUE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <ios>
# include <sstream>
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
# include <yampi/buffer.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/reduce.hpp>
# include <yampi/binary_operation.hpp>

# include <ket/spin_expectation_value.hpp>
# include <ket/qubit.hpp>
# include <ket/qubit_io.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/page/spin_expectation_value.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>

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
    namespace spin_expectation_value_detail
    {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
      template <typename ParallelPolicy, typename Qubit, typename Spin>
      struct call_spin_expectation_value
      {
        ParallelPolicy parallel_policy_;
        Qubit qubit_;
        Spin& spin_;

        call_spin_expectation_value(
          ParallelPolicy const parallel_policy, Qubit const qubit, Spin& spin)
          : parallel_policy_(parallel_policy),
            qubit_(qubit),
            spin_(spin)
        { }

        template <typename RandomAccessIterator>
        void operator()(
          RandomAccessIterator const first,
          RandomAccessIterator const last) const
        {
          Spin const local_spin
            = ::ket::spin_expectation_value(parallel_policy_, first, last, qubit_);
          spin_[0u] += local_spin[0u];
          spin_[1u] += local_spin[1u];
          spin_[2u] += local_spin[2u];
        }
      };

      template <typename ParallelPolicy, typename Qubit, typename Spin>
      inline call_spin_expectation_value<ParallelPolicy, Qubit, Spin> make_call_spin_expectation_value(
        ParallelPolicy const parallel_policy, Qubit const qubit, Spin& spin)
      {
        return call_spin_expectation_value<ParallelPolicy, Qubit, Spin>(
          parallel_policy, qubit, spin);
      }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
    } // namespace spin_expectation_value_detail

    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u> >::type
    spin_expectation_value(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      std::ostringstream output_string_stream("Spin ", std::ios_base::ate);
      output_string_stream << qubit;
      ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      KET_array<qubit_type, 1u> qubits = { qubit };
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation,
        buffer, complex_datatype, communicator, environment);

      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      spin_type spin = { };

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              mpi_policy, parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            spin_type const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      spin_type result;
      yampi::all_reduce(
        communicator, environment,
        yampi::make_buffer(::ket::utility::begin(spin), ::ket::utility::end(spin), real_datatype),
        ::ket::utility::begin(result), yampi::binary_operation(::yampi::plus_t()));

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u> >::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      KET_array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u> >::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, communicator, environment);
    }


    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    spin_expectation_value(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      std::ostringstream output_string_stream("Spin ", std::ios_base::ate);
      output_string_stream << qubit;
      ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      KET_array<qubit_type, 1u> qubits = { qubit };
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation,
        buffer, complex_datatype, communicator, environment);

      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      typedef KET_array<real_type, 3u> spin_type;
      spin_type spin = { };

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              mpi_policy, parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            spin_type const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      spin_type result;
      yampi::reduce(communicator, root).call(
        environment,
        yampi::make_buffer(::ket::utility::begin(spin), ::ket::utility::end(spin), real_datatype),
        ::ket::utility::begin(result), yampi::binary_operation(yampi::plus_t()));

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        KET_array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u> > >::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype const real_datatype,
      yampi::datatype const complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubit, permutation,
        buffer, real_datatype, complex_datatype, root, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


# undef KET_enable_if
# undef KET_array

#endif
