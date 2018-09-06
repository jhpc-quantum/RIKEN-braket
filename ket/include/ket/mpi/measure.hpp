#ifndef KET_MPI_MEASURE_HPP
# define KET_MPI_MEASURE_HPP

# include <boost/config.hpp>

# include <cmath>
# include <vector>

# include <boost/utility.hpp> // boost::prior

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# include <boost/range/adaptor/transformed.hpp>
# include <boost/range/algorithm/upper_bound.hpp>
# include <boost/range/numeric.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/buffer.hpp>
# include <yampi/gather.hpp>
# include <yampi/broadcast.hpp>
# include <yampi/basic_datatype_of.hpp>
# include <yampi/algorithm/ranked_buffer.hpp>
# include <yampi/algorithm/transform.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/positive_random_value_upto.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/fill.hpp>


namespace ket
{
  namespace mpi
  {
    // measure
    namespace measure_detail
    {
      template <typename Complex>
      struct complex_norm
      {
        typedef Complex result_type;

        Complex operator()(Complex const& value) const
        { using std::norm; return static_cast<Complex>(norm(value)); }
      };

# ifdef BOOST_NO_CXX11_LAMBDAS
      struct real_part_less_than
      {
        typedef bool result_type;

        template <typename Complex>
        bool operator()(Complex const& lhs, Complex const& rhs) const
        { using std::real;return real(lhs) < real(rhs); }
      };


      template <typename TotalProbabilities>
      struct modify_random_value
      {
       private:
        TotalProbabilities const& total_probabilities_;
        yampi::rank result_rank_;

       public:
        typedef typename TotalProbabilities::value_type result_type;

        modify_random_value(
          TotalProbabilities const& total_probabilities, yampi::rank const result_rank)
          : total_probabilities_(total_probabilities), result_rank_(result_rank)
        { }

        result_type operator()(result_type const random_value) const
        { return random_value - total_probabilities_[result_rank_.mpi_rank()-1]; }
      };

      template <typename TotalProbabilities>
      inline ::ket::mpi::measure_detail::modify_random_value<TotalProbabilities>
      make_modify_random_value(
        TotalProbabilities const& total_probabilities, yampi::rank const result_rank)
      {
        return ::ket::mpi::measure_detail::modify_random_value<TotalProbabilities>(
          total_probabilities, result_rank);
      }
# endif // BOOST_NO_CXX11_LAMBDAS
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger measure(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype const state_integer_datatype,
      yampi::datatype const real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ket::mpi::utility::log_with_time_guard<char> print("Measurement", environment);

      typedef typename boost::range_value<LocalState>::type complex_type;
      ::ket::utility::ranges::inclusive_scan(
        parallel_policy,
        local_state | boost::adaptors::transformed(
          ::ket::mpi::measure_detail::complex_norm<complex_type>()),
        boost::begin(local_state));

      yampi::rank const present_rank = communicator.rank(environment);
      BOOST_CONSTEXPR_OR_CONST yampi::rank root_rank(0);

      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      std::vector<real_type> total_probabilities;
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      using std::real;
      yampi::gather(communicator, root_rank).call(
        environment,
        yampi::make_buffer(
          real(*boost::prior(boost::end(local_state))), real_datatype),
        boost::begin(total_probabilities));

      real_type random_value;
      yampi::rank result_rank;
      if (present_rank == root_rank)
      {
        boost::partial_sum(total_probabilities, total_probabilities.begin());

        random_value
          = ::ket::utility::positive_random_value_upto(
              total_probabilities.back(), random_number_generator);
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              boost::size(boost::upper_bound<boost::return_begin_found>(
                total_probabilities, random_value))));
      }

      int result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(communicator, root_rank).call(
        environment, yampi::make_buffer(result_mpi_rank, yampi::basic_datatype_of<int>::call()));
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

# ifndef BOOST_NO_CXX11_LAMBDAS
      yampi::algorithm::transform(
        yampi::ignore_status(), communicator, environment,
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, root_rank),
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, result_rank),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank()-1]; });
# else
      yampi::algorithm::transform(
        yampi::ignore_status(), communicator, environment,
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, root_rank),
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, result_rank),
        ::ket::mpi::measure_detail::make_modify_random_value(total_probabilities, result_rank));
# endif

      StateInteger permutated_result;
      if (present_rank == result_rank)
      {
# ifndef BOOST_NO_CXX11_LAMBDAS
        StateInteger const local_result
          = static_cast<StateInteger>(
              boost::size(boost::upper_bound<boost::return_begin_found>(
                local_state, static_cast<complex_type>(random_value),
                [](complex_type const& lhs, complex_type const& rhs)
                { using std::real; return real(lhs) < real(rhs); })));
# else // BOOST_NO_CXX11_LAMBDAS
        StateInteger const local_result
          = static_cast<StateInteger>(
              boost::size(boost::upper_bound<boost::return_begin_found>(
                local_state, static_cast<complex_type>(random_value),
                ::ket::mpi::measure_detail::real_part_less_than())));
# endif // BOOST_NO_CXX11_LAMBDAS
        using ::ket::mpi::utility::rank_index_to_qubit_value;
        permutated_result
          = rank_index_to_qubit_value(
              mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type(real_type(0)));
        boost::begin(local_state)[local_result] = complex_type(real_type(1));
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type(real_type(0)));

      yampi::broadcast(communicator, result_rank).call(
        environment, yampi::make_buffer(permutated_result, state_integer_datatype));

      using ::ket::mpi::inverse_permutate_bits;
      return inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype const state_integer_datatype,
      yampi::datatype const real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::measure(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger measure(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      return ::ket::mpi::measure(
        mpi_policy, parallel_policy,
        local_state, random_number_generator, permutation,
        yampi::basic_datatype_of<real_type>::call(),
        yampi::basic_datatype_of<StateInteger>::call(),
        communicator, environment);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::measure(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        communicator, environment);
    }


    // fast_measure
    namespace measure_detail
    {
      template <typename Real>
      struct norm
      {
        typedef Real result_type;

        template <typename Complex>
        Real operator()(Complex const& value) const
        { using std::norm; return norm(value); }
      };
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger fast_measure(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype const state_integer_datatype,
      yampi::datatype const real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      ket::mpi::utility::log_with_time_guard<char> print("Measurement (fast)", environment);

      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      std::vector<real_type> partial_sum_probabilities(boost::size(local_state), real_type(0));
      ::ket::utility::ranges::inclusive_scan(
        parallel_policy,
        local_state | boost::adaptors::transformed(
          ::ket::mpi::measure_detail::norm<real_type>()),
        partial_sum_probabilities.begin());

      yampi::rank const present_rank = communicator.rank(environment);
      BOOST_CONSTEXPR_OR_CONST yampi::rank root_rank(0);

      std::vector<real_type> total_probabilities;
      if (present_rank == root_rank)
        total_probabilities.resize(communicator.size(environment));

      using std::real;
      yampi::gather(communicator, root_rank).call(
        environment,
        yampi::make_buffer(
          partial_sum_probabilities.back(), real_datatype),
        boost::begin(total_probabilities));

      real_type random_value;
      yampi::rank result_rank;
      if (present_rank == root_rank)
      {
        boost::partial_sum(total_probabilities, total_probabilities.begin());

        random_value
          = ::ket::utility::positive_random_value_upto(
              total_probabilities.back(), random_number_generator);
        result_rank
          = static_cast<yampi::rank>(static_cast<StateInteger>(
              boost::size(boost::upper_bound<boost::return_begin_found>(
                total_probabilities, random_value))));
      }

      int result_mpi_rank = result_rank.mpi_rank();
      yampi::broadcast(communicator, root_rank).call(
        environment, yampi::make_buffer(result_mpi_rank, yampi::basic_datatype_of<int>::call()));
      result_rank = static_cast<yampi::rank>(result_mpi_rank);

# ifndef BOOST_NO_CXX11_LAMBDAS
      yampi::algorithm::transform(
        yampi::ignore_status(), communicator, environment,
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, root_rank),
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, result_rank),
        [&total_probabilities, result_rank](real_type const random_value)
        { return random_value - total_probabilities[result_rank.mpi_rank()-1]; });
# else
      yampi::algorithm::transform(
        yampi::ignore_status(), communicator, environment,
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, root_rank),
        yampi::algorithm::make_ranked_buffer(random_value, real_datatype, result_rank),
        ::ket::mpi::measure_detail::make_modify_random_value(total_probabilities, result_rank));
# endif

      StateInteger permutated_result;
      if (present_rank == result_rank)
      {
        StateInteger const local_result
          = static_cast<StateInteger>(
              boost::size(boost::upper_bound<boost::return_begin_found>(
                partial_sum_probabilities, random_value)));
        using ::ket::mpi::utility::rank_index_to_qubit_value;
        permutated_result
          = rank_index_to_qubit_value(
              mpi_policy, local_state, result_rank, local_result);

        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type(real_type(0)));
        boost::begin(local_state)[local_result] = complex_type(real_type(1));
      }
      else
        ::ket::mpi::utility::fill(
          mpi_policy, parallel_policy, local_state, complex_type(real_type(0)));

      yampi::broadcast(communicator, result_rank).call(
        environment, yampi::make_buffer(permutated_result, state_integer_datatype));

      using ::ket::mpi::inverse_permutate_bits;
      return inverse_permutate_bits(permutation, permutated_result);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger fast_measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::datatype const state_integer_datatype,
      yampi::datatype const real_datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fast_measure(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        state_integer_datatype, real_datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger fast_measure(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<LocalState>::type complex_type;
      typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
      return ::ket::mpi::fast_measure(
        mpi_policy, parallel_policy,
        local_state, random_number_generator, permutation,
        yampi::basic_datatype_of<real_type>::call(),
        yampi::basic_datatype_of<StateInteger>::call(),
        communicator, environment);
    }

    template <
      typename LocalState, typename RandomNumberGenerator,
      typename StateInteger, typename BitInteger, typename Allocator>
    inline StateInteger fast_measure(
      LocalState& local_state, RandomNumberGenerator& random_number_generator,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::fast_measure(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, random_number_generator, permutation,
        communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif

