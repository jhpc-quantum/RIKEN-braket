#ifndef KET_MPI_EXPECTATION_VALUE_HPP
# define KET_MPI_EXPECTATION_VALUE_HPP

# include <array>
# include <vector>
# include <string>
# include <iterator>
# include <numeric>
# include <utility>
# include <type_traits>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/reduce.hpp>
# include <yampi/binary_operation.hpp>

# include <ket/expectation_value.hpp>
# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/page/none_on_page.hpp>
# include <ket/mpi/gate/gate.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    namespace local
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto expectation_value(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >
      {
        ::ket::mpi::gate::detail::assert_all_qubits_are_local(
          mpi_policy, local_state, communicator, environment, permutated_qubit, permutated_qubits...);

        constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 1u);
        using real_type = ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> >;

        struct { Observable call; } wrapped_observable{std::forward<Observable>(observable)};
        if (::ket::mpi::page::none_on_page(local_state, permutated_qubit, permutated_qubits...))
        {
          auto result = real_type{0};
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, wrapped_observable = std::move(wrapped_observable), permutated_qubit, permutated_qubits..., &result](auto const first, auto const last)
            { result += ::ket::expectation_value(parallel_policy, first, last, wrapped_observable.call, permutated_qubit.qubit(), permutated_qubits.qubit()...); });

          return result;
        }

        auto partial_sums = std::vector<real_type>(::ket::utility::num_threads(parallel_policy));
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        ::ket::mpi::gate::local::gate(
          mpi_policy, parallel_policy,
          local_state, buffer, communicator, environment,
          [wrapped_observable = std::move(wrapped_observable), &partial_sums](
            auto const first, StateInteger const index_wo_qubits,
            std::array<qubit_type, num_operated_qubits> const& unsorted_qubits,
            std::array<qubit_type, num_operated_qubits + 1u> const& sorted_qubits_with_sentinel,
            int const thread_index)
          { partial_sums[thread_index] += wrapped_observable.call(first, index_wo_qubits, unsorted_qubits, sorted_qubits_with_sentinel); },
          permutated_qubit, permutated_qubits...);
# else // KET_USE_BIT_MASKS_EXPLICITLY
        ::ket::mpi::gate::local::gate(
          mpi_policy, parallel_policy,
          local_state, buffer, communicator, environment,
          [wrapped_observable = std::move(wrapped_observable), &partial_sums](
            auto const first, StateInteger const index_wo_qubits,
            std::array<StateInteger, num_operated_qubits> const& qubit_masks,
            std::array<StateInteger, num_operated_qubits + 1u> const& index_masks,
            int const thread_index)
          { partial_sums[thread_index] += wrapped_observable.call(first, index_wo_qubits, qubit_masks, index_masks); },
          permutated_qubit, permutated_qubits...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using std::begin;
        using std::end;
        return std::accumulate(begin(partial_sums), end(partial_sums), real_type{0});
      }
    } // namespace local

    // all_reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Expectation value for qubits "}, qubit, qubits...), environment};

      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment, qubit, qubits...);

      auto result
        = ::ket::mpi::local::expectation_value(
            mpi_policy, parallel_policy,
            local_state, buffer, communicator, environment,
            std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

      yampi::all_reduce(
        yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
        communicator, environment);

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Expectation value for qubits "}, qubit, qubits...), environment};

      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, complex_datatype, communicator, environment, qubit, qubits...);

      auto result
        = ::ket::mpi::local::expectation_value(
            mpi_policy, parallel_policy,
            local_state, buffer, communicator, environment,
            std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

      yampi::all_reduce(
        yampi::in_place, yampi::make_buffer(result, real_datatype), yampi::binary_operation{::yampi::tags::plus},
        communicator, environment);

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer,
        real_datatype, complex_datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > >
    expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer,
        real_datatype, complex_datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }


    // reduce version
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Expectation value for qubits "}, qubit, qubits...), environment};

      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment, qubit, qubits...);

      auto result
        = ::ket::mpi::local::expectation_value(
            mpi_policy, parallel_policy,
            local_state, buffer, communicator, environment,
            std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

      yampi::reduce(
        yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
        root, communicator, environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Expectation value for qubits "}, qubit, qubits...), environment};

      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, complex_datatype, communicator, environment, qubit, qubits...);

      auto result
        = ::ket::mpi::local::expectation_value(
            mpi_policy, parallel_policy,
            local_state, buffer, communicator, environment,
            std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

      yampi::reduce(
        yampi::in_place, yampi::make_buffer(result, real_datatype), yampi::binary_operation{::yampi::tags::plus},
        root, communicator, environment);

      if (communicator.rank(environment) != root)
        return boost::none;

      return result;
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, real_datatype, complex_datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator, typename DerivedDatatype1, typename DerivedDatatype2,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value,
      boost::optional< ::ket::utility::meta::real_t< ::ket::utility::meta::range_value_t<LocalState> > > >
    expectation_value(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype1> const& real_datatype,
      yampi::datatype_base<DerivedDatatype2> const& complex_datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::expectation_value(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, real_datatype, complex_datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_EXPECTATION_VALUE_HPP
