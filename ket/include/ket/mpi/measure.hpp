#ifndef KET_MPI_MEASURE_HPP
# define KET_MPI_MEASURE_HPP

# include <cmath>
# include <vector>
# include <iterator>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/buffer.hpp>
# include <yampi/gather.hpp>
# include <yampi/broadcast.hpp>
# include <yampi/message_envelope.hpp>
# include <yampi/algorithm/transform.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/fill.hpp>
# include <ket/mpi/utility/transform_inclusive_scan.hpp>
# include <ket/mpi/utility/transform_inclusive_scan_self.hpp>
# include <ket/mpi/utility/upper_bound.hpp>


namespace ket
{
  namespace mpi
  {
    // measure
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline auto measure(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Measurement", environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using std::real;
      auto const total_probability
        = real(::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy, local_state,
            [](complex_type const& lhs, complex_type const& rhs)
            { using std::real; return static_cast<complex_type>(real(lhs) + real(rhs)); },
            [](complex_type const& value)
            { using std::norm; return static_cast<complex_type>(norm(value)); },
            environment));

      auto const present_rank = communicator.rank(environment);
      constexpr auto root_rank = yampi::rank{0};

      using real_type = ::ket::utility::meta::real_t<complex_type>;
      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      using std::begin;
      yampi::gather(
        yampi::make_buffer(total_probability), begin(total_probabilities),
        root_rank, communicator, environment);

      auto random_value = real_type{};
      auto result_rank = yampi::rank{};
      if (present_rank == root_rank)
      {
        ::ket::utility::ranges::inclusive_scan(total_probabilities, begin(total_probabilities));

        random_value
          = ::ket::utility::positive_random_value_upto(
              total_probabilities.back(), random_number_generator);
        using std::end;
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              std::upper_bound(begin(total_probabilities), end(total_probabilities), random_value)
              - begin(total_probabilities)));
      }

      auto result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(yampi::make_buffer(result_mpi_rank), root_rank, communicator, environment);
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

      yampi::algorithm::transform(
        yampi::ignore_status,
        yampi::make_buffer(random_value),
        yampi::make_buffer(random_value),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank() - 1]; },
        ::yampi::message_envelope(root_rank, result_rank, communicator),
        environment);

      auto permutated_result = StateInteger{};
      if (present_rank == result_rank)
      {
        auto const local_result
          = static_cast<StateInteger>(
              ::ket::mpi::utility::upper_bound(
                local_state, static_cast<complex_type>(random_value),
                [](complex_type const& lhs, complex_type const& rhs)
                { using std::real; return real(lhs) < real(rhs); },
                environment));

        permutated_result
          = ::ket::mpi::utility::rank_index_to_qubit_value(mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);
        begin(local_state)[local_result] = complex_type{real_type{1}};
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);

      yampi::broadcast(yampi::make_buffer(permutated_result), result_rank, communicator, environment);

      return ::ket::mpi::inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline auto measure(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    -> StateInteger
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Measurement", environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using std::real;
      auto const total_probability
        = real(::ket::mpi::utility::transform_inclusive_scan_self(
            parallel_policy, local_state,
            [](complex_type const& lhs, complex_type const& rhs)
            { using std::real; return static_cast<complex_type>(real(lhs) + real(rhs)); },
            [](complex_type const& value)
            { using std::norm; return static_cast<complex_type>(norm(value)); },
            environment));

      auto const present_rank = communicator.rank(environment);
      constexpr auto root_rank = yampi::rank{0};

      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      using std::begin;
      yampi::gather(
        yampi::make_buffer(total_probability, real_datatype), begin(total_probabilities),
        root_rank, communicator, environment);

      auto random_value = real_type{};
      auto result_rank = yampi::rank{};
      if (present_rank == root_rank)
      {
        ::ket::utility::ranges::inclusive_scan(
          total_probabilities, begin(total_probabilities));

        random_value
          = ::ket::utility::positive_random_value_upto(total_probabilities.back(), random_number_generator);
        using std::end;
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              std::upper_bound(begin(total_probabilities), end(total_probabilities), random_value)
              - begin(total_probabilities)));
      }

      auto result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(yampi::make_buffer(result_mpi_rank), root_rank, communicator, environment);
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

      yampi::algorithm::transform(
        yampi::ignore_status,
        yampi::make_buffer(random_value, real_datatype),
        yampi::make_buffer(random_value, real_datatype),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank() - 1]; },
        ::yampi::message_envelope(root_rank, result_rank, communicator),
        environment);

      auto permutated_result = StateInteger{};
      if (present_rank == result_rank)
      {
        auto const local_result
          = static_cast<StateInteger>(
              ::ket::mpi::utility::upper_bound(
                local_state, static_cast<complex_type>(random_value),
                [](complex_type const& lhs, complex_type const& rhs)
                { using std::real; return real(lhs) < real(rhs); },
                environment));

        permutated_result
          = ::ket::mpi::utility::rank_index_to_qubit_value(mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);
        begin(local_state)[local_result] = complex_type{real_type{1}};
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);

      yampi::broadcast(yampi::make_buffer(permutated_result, state_integer_datatype), result_rank, communicator, environment);

      return ::ket::mpi::inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline auto measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      return ::ket::mpi::measure(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation, communicator, environment);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline auto measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      return ::ket::mpi::measure(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }


    // fast_measure
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline auto fast_measure(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Measurement (fast)", environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using std::begin;
      using std::end;
      auto partial_sum_probabilities = std::vector<real_type>(std::distance(begin(local_state), end(local_state)), real_type{0});
      ::ket::mpi::utility::transform_inclusive_scan(
        parallel_policy, local_state, begin(partial_sum_probabilities),
        [](real_type const& lhs, real_type const& rhs) { return lhs + rhs; },
        [](complex_type const& value) { using std::norm; return norm(value); },
        environment);

      auto const present_rank = communicator.rank(environment);
      constexpr auto root_rank = yampi::rank{0};

      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      yampi::gather(
        yampi::make_buffer(partial_sum_probabilities.back()), begin(total_probabilities),
        root_rank, communicator, environment);

      auto random_value = real_type{};
      auto result_rank = yampi::rank{};
      if (present_rank == root_rank)
      {
        ::ket::utility::ranges::inclusive_scan(total_probabilities, begin(total_probabilities));

        random_value
          = ::ket::utility::positive_random_value_upto(total_probabilities.back(), random_number_generator);
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              std::upper_bound(begin(total_probabilities), end(total_probabilities), random_value)
              - begin(total_probabilities)));
      }

      auto result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(yampi::make_buffer(result_mpi_rank), root_rank, communicator, environment);
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

      yampi::algorithm::transform(
        yampi::ignore_status,
        yampi::make_buffer(random_value),
        yampi::make_buffer(random_value),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank() - 1]; },
        ::yampi::message_envelope(root_rank, result_rank, communicator),
        environment);

      auto permutated_result = StateInteger{};
      if (present_rank == result_rank)
      {
        auto const local_result
          = static_cast<StateInteger>(
              std::upper_bound(begin(partial_sum_probabilities), end(partial_sum_probabilities), random_value)
              - begin(partial_sum_probabilities));
        permutated_result
          = ::ket::mpi::utility::rank_index_to_qubit_value(mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);
        begin(local_state)[local_result] = complex_type{real_type{1}};
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);

      yampi::broadcast(yampi::make_buffer(permutated_result), result_rank, communicator, environment);

      return ::ket::mpi::inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline auto fast_measure(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Measurement (fast)", environment};

      using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
      using real_type = ::ket::utility::meta::real_t<complex_type>;
      using std::begin;
      using std::end;
      auto partial_sum_probabilities = std::vector<real_type>(std::distance(begin(local_state), end(local_state)), real_type{0});
      ::ket::mpi::utility::transform_inclusive_scan(
        parallel_policy, local_state, begin(partial_sum_probabilities),
        [](real_type const& lhs, real_type const& rhs) { return lhs + rhs; },
        [](complex_type const& value) { using std::norm; return norm(value); },
        environment);

      auto const present_rank = communicator.rank(environment);
      constexpr auto root_rank = yampi::rank{0};

      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      using std::real;
      yampi::gather(
        yampi::make_buffer(partial_sum_probabilities.back(), real_datatype), begin(total_probabilities),
        root_rank, communicator, environment);

      auto random_value = real_type{};
      auto result_rank = yampi::rank{};
      if (present_rank == root_rank)
      {
        ::ket::utility::ranges::inclusive_scan(total_probabilities, begin(total_probabilities));

        random_value
          = ::ket::utility::positive_random_value_upto(total_probabilities.back(), random_number_generator);
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              std::upper_bound(begin(total_probabilities), end(total_probabilities), random_value)
              - begin(total_probabilities)));
      }

      auto result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(yampi::make_buffer(result_mpi_rank), root_rank, communicator, environment);
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

      yampi::algorithm::transform(
        yampi::ignore_status,
        yampi::make_buffer(random_value, real_datatype),
        yampi::make_buffer(random_value, real_datatype),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank() - 1]; },
        ::yampi::message_envelope(root_rank, result_rank, communicator),
        environment);

      auto permutated_result = StateInteger{};
      if (present_rank == result_rank)
      {
        auto const local_result
          = static_cast<StateInteger>(
              std::upper_bound(begin(partial_sum_probabilities), end(partial_sum_probabilities), random_value)
              - begin(partial_sum_probabilities));
        permutated_result
          = ::ket::mpi::utility::rank_index_to_qubit_value(mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);
        begin(local_state)[local_result] = complex_type{real_type{1}};
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type{real_type{0}}, communicator, environment);

      yampi::broadcast(yampi::make_buffer(permutated_result, state_integer_datatype), result_rank, communicator, environment);

      return ::ket::mpi::inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline auto fast_measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      return ::ket::mpi::fast_measure(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation, communicator, environment);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline auto fast_measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    -> StateInteger
    {
      return ::ket::mpi::fast_measure(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_MEASURE_HPP
