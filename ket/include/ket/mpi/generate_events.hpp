#ifndef KET_MPI_GENERATE_EVENTS_HPP
# define KET_MPI_GENERATE_EVENTS_HPP

# include <cmath>
# include <vector>
# include <iterator>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>

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
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/fill.hpp>
# include <ket/mpi/utility/transform_inclusive_scan_self.hpp>
# include <ket/mpi/utility/upper_bound.hpp>


namespace ket
{
  namespace mpi
  {
    // generate_events
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline void generate_events(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Generate Events", environment};

      result.clear();
      result.reserve(num_events);

      using complex_type = typename boost::range_value<LocalState>::type;
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

      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      yampi::gather(root_rank, communicator).call(
        yampi::make_buffer(total_probability),
        std::begin(total_probabilities), environment);

      if (present_rank == root_rank)
        ::ket::utility::ranges::inclusive_scan(
          total_probabilities, std::begin(total_probabilities));

      for (auto event_index = 0; event_index < num_events; ++event_index)
      {
        auto random_value = real_type{};
        auto result_rank = yampi::rank{};
        if (present_rank == root_rank)
        {
          random_value
            = ::ket::utility::positive_random_value_upto(
                total_probabilities.back(), random_number_generator);
          result_rank
            = static_cast<yampi::rank>(static_cast<StateInteger>(
                std::upper_bound(
                  std::begin(total_probabilities),
                  std::end(total_probabilities), random_value)
                - std::begin(total_probabilities)));
        }

        auto result_mpi_rank = result_rank.mpi_rank();
        yampi::broadcast(root_rank, communicator).call(
          yampi::make_buffer(result_mpi_rank), environment);
        result_rank = static_cast<yampi::rank>(result_mpi_rank);

        yampi::algorithm::transform(
          yampi::ignore_status(),
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

          using ::ket::mpi::utility::rank_index_to_qubit_value;
          permutated_result
            = rank_index_to_qubit_value(
                mpi_policy, local_state, result_rank, local_result);
        }

        yampi::broadcast(result_rank, communicator).call(
          yampi::make_buffer(permutated_result), environment);

        using ::ket::mpi::inverse_permutate_bits;
        result.push_back(inverse_permutate_bits(permutation, permutated_result));
      }
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline void generate_events(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ket::mpi::utility::log_with_time_guard<char> print{"Generate Events", environment};

      result.clear();
      result.reserve(num_events);

      using complex_type = typename boost::range_value<LocalState>::type;
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

      using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
      auto total_probabilities = std::vector<real_type>{};
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      yampi::gather(root_rank, communicator).call(
        yampi::make_buffer(total_probability, real_datatype),
        std::begin(total_probabilities),
        environment);

      if (present_rank == root_rank)
        ::ket::utility::ranges::inclusive_scan(
          total_probabilities, std::begin(total_probabilities));

      for (auto event_index = 0; event_index < num_events; ++event_index)
      {
        auto random_value = real_type{};
        auto result_rank = yampi::rank{};
        if (present_rank == root_rank)
        {
          random_value
            = ::ket::utility::positive_random_value_upto(
                total_probabilities.back(), random_number_generator);
          result_rank
            = static_cast<yampi::rank>(static_cast<StateInteger>(
                std::upper_bound(
                  std::begin(total_probabilities),
                  std::end(total_probabilities), random_value)
                - std::begin(total_probabilities)));
        }

        auto result_mpi_rank = result_rank.mpi_rank();
        yampi::broadcast(root_rank, communicator).call(
          yampi::make_buffer(result_mpi_rank),
          environment);
        result_rank = static_cast<yampi::rank>(result_mpi_rank);

        yampi::algorithm::transform(
          yampi::ignore_status(),
          yampi::make_buffer(random_value, real_datatype),
          yampi::make_buffer(random_value, real_datatype),
          [&total_probabilities, result_rank](real_type const random_value)
          { return random_value - total_probabilities[result_rank.mpi_rank() - 1]; },
          ::yampi::message_envelope(root_rank, result_rank, communicator),
          environment);

        StateInteger permutated_result;
        if (present_rank == result_rank)
        {
          StateInteger const local_result
            = static_cast<StateInteger>(
                ::ket::mpi::utility::upper_bound(
                  local_state, static_cast<complex_type>(random_value),
                  [](complex_type const& lhs, complex_type const& rhs)
                  { using std::real; return real(lhs) < real(rhs); },
                  environment));

          using ::ket::mpi::utility::rank_index_to_qubit_value;
          permutated_result
            = rank_index_to_qubit_value(
                mpi_policy, local_state, result_rank, local_result);
        }

        yampi::broadcast(result_rank, communicator).call(
          yampi::make_buffer(permutated_result, state_integer_datatype), environment);

        using ::ket::mpi::inverse_permutate_bits;
        result.push_back(inverse_permutate_bits(permutation, permutated_result));
      }
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline void generate_events(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator const&,
      typename RandomNumberGenerator::result_type const seed,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      RandomNumberGenerator random_number_generator(seed);
      ::ket::mpi::generate_events(
        mpi_policy, parallel_policy,
        result, local_state, num_events, random_number_generator, permutation,
        communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline void generate_events(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator const&,
      typename RandomNumberGenerator::result_type const seed,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      RandomNumberGenerator random_number_generator(seed);
      ::ket::mpi::generate_events(
        mpi_policy, parallel_policy,
        result, local_state, num_events, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }

    template <
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline void generate_events(
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::generate_events(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        result, local_state, num_events, random_number_generator, permutation,
        communicator, environment);
    }

    template <
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline void generate_events(
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::generate_events(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        result, local_state, num_events, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }

    template <
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline void generate_events(
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator const&,
      typename RandomNumberGenerator::result_type const seed,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      RandomNumberGenerator random_number_generator(seed);
      ::ket::mpi::generate_events(
        result, local_state, num_events, random_number_generator, permutation,
        communicator, environment);
    }

    template <
      typename ResultAllocator,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename DerivedDatatype1, typename DerivedDatatype2>
    inline void generate_events(
      std::vector<StateInteger, ResultAllocator>& result,
      LocalState& local_state,
      int const num_events,
      RandomNumberGenerator const&,
      typename RandomNumberGenerator::result_type const seed,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype_base<DerivedDatatype1> const& state_integer_datatype,
      yampi::datatype_base<DerivedDatatype2> const& real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      RandomNumberGenerator random_number_generator(seed);
      ::ket::mpi::generate_events(
        result, local_state, num_events, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GENERATE_EVENTS_HPP
