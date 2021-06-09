#ifndef KET_MPI_SPIN_EXPECTATION_VALUE_HPP
# define KET_MPI_SPIN_EXPECTATION_VALUE_HPP

# include <boost/config.hpp>

# include <array>
# include <ios>
# include <sstream>
# include <iterator>
# include <type_traits>

# include <boost/optional.hpp>

# include <boost/range/value_type.hpp>

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
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/page/spin_expectation_value.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>


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
          auto const local_spin
            = ::ket::spin_expectation_value(parallel_policy_, first, last, qubit_);
          spin_[0u] += local_spin[0u];
          spin_[1u] += local_spin[1u];
          spin_[2u] += local_spin[2u];
        }
      }; // struct call_spin_expectation_value<ParallelPolicy, Qubit, Spin>

      template <typename ParallelPolicy, typename Qubit, typename Spin>
      inline call_spin_expectation_value<ParallelPolicy, Qubit, Spin> make_call_spin_expectation_value(
        ParallelPolicy const parallel_policy, Qubit const qubit, Spin& spin)
      {
        return call_spin_expectation_value<ParallelPolicy, Qubit, Spin>{
          parallel_policy, qubit, spin};
      }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
    } // namespace spin_expectation_value_detail

    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, communicator, environment);

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      auto result = spin_type{};
      yampi::all_reduce(
        yampi::make_buffer(std::begin(spin), std::end(spin)),
        std::begin(result), yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, complex_datatype, communicator, environment);

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      auto result = spin_type{};
      yampi::all_reduce(
        yampi::make_buffer(std::begin(spin), std::end(spin), real_datatype),
        std::begin(result), yampi::binary_operation(::yampi::plus_t()),
        communicator, environment);

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation, buffer, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubit, permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      std::array<
        typename ::ket::utility::meta::real_of<
          typename boost::range_value<LocalState>::type>::type, 3u>>::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
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
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, communicator, environment);

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      auto result = spin_type{};
      yampi::reduce(root, communicator).call(
        yampi::make_buffer(std::begin(spin), std::end(spin)),
        std::begin(result), yampi::binary_operation(yampi::plus_t()),
        environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Spin "}, qubit), environment};

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      auto qubits = std::array<qubit_type, 1u>{qubit};
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation, buffer, complex_datatype, communicator, environment);

      using complex_type = typename boost::range_value<LocalState>::type;
      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      using spin_type = std::array<real_type, 3u>;
      auto spin = spin_type{};

      if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
        spin
          = ::ket::mpi::page::spin_expectation_value(
              parallel_policy, local_state, qubit, permutation);
      else
      {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          [parallel_policy, qubit, &permutation, &spin](auto const first, auto const last)
          {
            auto const local_spin
              = ::ket::spin_expectation_value(parallel_policy, first, last, permutation[qubit]);
            spin[0u] += local_spin[0u];
            spin[1u] += local_spin[1u];
            spin[2u] += local_spin[2u];
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, communicator, environment,
          ::ket::mpi::spin_expectation_value_detail::make_call_spin_expectation_value(
            parallel_policy, permutation[qubit], spin));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      auto result = spin_type{};
      yampi::reduce(root, communicator).call(
        yampi::make_buffer(std::begin(spin), std::end(spin), real_datatype),
        std::begin(result), yampi::binary_operation(yampi::plus_t()),
        environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubit, permutation, buffer, root, communicator, environment);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::rank const root,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::spin_expectation_value(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubit, permutation, buffer, root, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional<
        std::array<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<LocalState>::type>::type, 3u>>>::type
    spin_expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::qubit<StateInteger, BitInteger> const qubit,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
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


#endif // KET_MPI_SPIN_EXPECTATION_VALUE_HPP
