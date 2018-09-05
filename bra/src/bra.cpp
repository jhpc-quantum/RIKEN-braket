#include <boost/config.hpp>

#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#ifndef BOOST_NO_CXX11_HDR_RANDOM
# include <random>
#else
# include <boost/random/mersenne_twister.hpp>
#endif
#ifndef BOOST_NO_CXX11_HDR_CHRONO
#  include <chrono>
#else
#  define BOOST_CHRONO_HEADER_ONLY
#  include <boost/chrono/chrono.hpp>
#endif

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <boost/move/unique_ptr.hpp>

#ifndef BRA_NO_MPI
# include <yampi/environment.hpp>
# include <yampi/thread_support.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/wall_clock.hpp>
#endif

#include <ket/utility/integer_exp2.hpp>
#include <ket/utility/integer_log2.hpp>

#include <bra/gates.hpp>
#include <bra/state.hpp>
#ifndef BRA_NO_MPI
# include <bra/make_general_mpi_state.hpp>
#else
# include <bra/nompi_state.hpp>
#endif

#ifndef BOOST_NO_CXX11_HDR_RANDOM
# define BRA_mt19937_64 std::mt19937_64
#else
# define BRA_mt19937_64 boost::random::mt19937_64
#endif

#ifndef BOOST_NO_CXX11_HDR_CHRONO
# define BRA_chrono std::chrono
#else // BOOST_NO_CXX11_HDR_CHRONO
# define BRA_chrono boost::chrono
#endif // BOOST_NO_CXX11_HDR_CHRONO

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else // BRA_NO_MPI
# define BRA_clock BRA_chrono::system_clock
#endif // BRA_NO_MPI


template <typename StateInteger, typename BitInteger>
std::string integer_to_bits_string(StateInteger const integer, BitInteger const total_num_qubits)
{
  std::string result;
  result.reserve(total_num_qubits);

  for (BitInteger left_bit = total_num_qubits; left_bit > static_cast<BitInteger>(0u); --left_bit)
  {
    StateInteger const zero_or_one = (integer bitand (static_cast<StateInteger>(1u) << (left_bit-1u))) >> (left_bit-1u);
    if (zero_or_one == static_cast<StateInteger>(0u))
      result.push_back('0');
    else
      result.push_back('1');
  }
  return result;
}

template <typename Clock, typename Duration>
double duration_to_second(
  BRA_chrono::time_point<Clock, Duration> const& from,
  BRA_chrono::time_point<Clock, Duration> const& to)
{ return 0.000001 * BRA_chrono::duration_cast<BRA_chrono::microseconds>(to - from).count(); }


int main(int argc, char* argv[])
{
  std::ios::sync_with_stdio(false);

  typedef unsigned int bit_integer_type;
  typedef BRA_mt19937_64 rng_type;
  typedef rng_type::result_type seed_type;

#ifndef BRA_NO_MPI
  yampi::environment environment(argc, argv, yampi::thread_support::funneled);
  yampi::world_communicator_t const world_communicator_tag;
  yampi::communicator communicator(world_communicator_tag);
  yampi::rank const rank = communicator.rank(environment);
  BOOST_CONSTEXPR_OR_CONST yampi::rank root_rank(0);
  bool const is_io_root_rank = rank == root_rank and yampi::is_io_process(root_rank, environment);

  if (environment.thread_support() == yampi::thread_support::single)
  {
    if (is_io_root_rank)
      std::cerr << "multithread environment is required" << std::endl;
    return EXIT_FAILURE;
  }

  bit_integer_type const num_gqubits
    = ket::utility::integer_log2<bit_integer_type>(communicator.size(environment));

  if (ket::utility::integer_exp2<bit_integer_type>(num_gqubits)
      != static_cast<bit_integer_type>(communicator.size(environment)))
  {
    if (is_io_root_rank)
      std::cerr << "wrong number of MPI processes" << std::endl;
    return EXIT_FAILURE;
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  if (argc < 2 or argc > 4)
  {
    if (is_io_root_rank)
      std::cerr << "wrong number of arguments: bra qcxfile [num_page_qubits [seed]]" << std::endl;
    return EXIT_FAILURE;
  }
#else // BRA_NO_MPI
  if (argc < 2 or argc > 3)
  {
    std::cerr << "wrong number of arguments: bra qcxfile [seed]" << std::endl;
    return EXIT_FAILURE;
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  std::string const filename(argv[1]);
  unsigned int const num_page_qubits
    = argc >= 3
      ? boost::lexical_cast<unsigned int>(argv[2])
      : 2u;
  seed_type const seed
    = argc == 4
      ? boost::lexical_cast<seed_type>(argv[3])
      : static_cast<seed_type>(1);
#else // BRA_NO_MPI
  std::string const filename(argv[1]);
  seed_type const seed
    = argc == 3
      ? boost::lexical_cast<seed_type>(argv[2])
      : static_cast<seed_type>(1);
#endif // BRA_NO_MPI

  std::ifstream file_stream(filename.c_str());
  if (!file_stream)
  {
#ifndef BRA_NO_MPI
    if (is_io_root_rank)
      std::cerr << "cannot open an input file " << filename << std::endl;
#else // BRA_NO_MPI
    std::cerr << "cannot open an input file " << filename << std::endl;
#endif // BRA_NO_MPI
    return EXIT_FAILURE;
  }

#ifndef BRA_NO_MPI
  bra::gates gates(file_stream, environment, root_rank, communicator);
  boost::movelib::unique_ptr<bra::state> state_ptr
    = bra::make_general_mpi_state(
        num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), gates.initial_permutation(),
        seed, communicator, environment);
#else // BRA_NO_MPI
  bra::gates gates(file_stream);
  boost::movelib::unique_ptr<bra::state> state_ptr
    = bra::make_nompi_state(gates.initial_state_value(), gates.num_qubits(), seed);
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  BRA_clock::time_point const start_time = BRA_clock::now(environment);
#else
  BRA_clock::time_point const start_time = BRA_clock::now();
#endif
  BRA_clock::time_point last_processed_time = start_time;

  *state_ptr << gates;

#ifndef BRA_NO_MPI
  if (not is_io_root_rank)
    return EXIT_SUCCESS;
#endif

  std::size_t const num_finish_processes = state_ptr->num_finish_processes();
  for (std::size_t index = 0u; index < num_finish_processes; ++index)
  {
    typedef bra::state::time_and_process_type time_and_process_type;
    time_and_process_type finish_time_and_process = state_ptr->finish_time_and_process(index);

    if (finish_time_and_process.second == BRA_FINISHED_PROCESS_VALUE(operations))
    {
      std::cout
        << "Operations finished: "
        << duration_to_second(start_time, finish_time_and_process.first)
        << " ("
        << duration_to_second(last_processed_time, finish_time_and_process.first)
        << ')'
        << std::endl;
      last_processed_time = finish_time_and_process.first;
    }
    else if (finish_time_and_process.second == BRA_FINISHED_PROCESS_VALUE(begin_measurement))
    {
      std::cout
        << boost::format("Expectation values of spins:\n%|=8s| %|=8s| %|=8s|\n")
           % "<Qx>" % "<Qy>" % "<Qz>";
# ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
      typedef bra::state::spin_type spin_type;
      for (spin_type const& spin: *(state_ptr->maybe_expectation_values()))
        std::cout
          << boost::format("%|=8.3f| %|=8.3f| %|=8.3f|\n")
             % (0.5-spin[0u]) % (0.5-spin[1u]) % (0.5-spin[2u]);
# else // BOOST_NO_CXX11_RANGE_BASED_FOR
      typedef bra::state::spins_type::const_iterator const_iterator;
      const_iterator const last = state_ptr->maybe_expectation_values()->end();
      for (const_iterator iter = state_ptr->maybe_expectation_values()->begin();
           iter != last; ++iter)
        std::cout
          << boost::format("%|=8.3f| %|=8.3f| %|=8.3f|\n")
             % (0.5-(*iter)[0u]) % (0.5-(*iter)[1u]) % (0.5-(*iter)[2u]);
# endif // BOOST_NO_CXX11_RANGE_BASED_FOR
      std::cout << std::flush;

      std::cout
        << "Expectation values finished: "
        << duration_to_second(start_time, finish_time_and_process.first)
        << " ("
        << duration_to_second(last_processed_time, finish_time_and_process.first)
        << ')'
        << std::endl;
      last_processed_time = finish_time_and_process.first;
    }
    else if (finish_time_and_process.second == BRA_FINISHED_PROCESS_VALUE(ket_measure))
    {
      std::cout
        << "Measurement result: " << state_ptr->measured_value()
        << "\nMeasurement finished: "
        << duration_to_second(start_time, finish_time_and_process.first)
        << " ("
        << duration_to_second(last_processed_time, finish_time_and_process.first)
        << ')'
        << std::endl;
      break;
    }
    else if (finish_time_and_process.second == BRA_FINISHED_PROCESS_VALUE(generate_events))
    {
      std::cout << "Events:\n";
      std::size_t const num_events = state_ptr->generated_events().size();
      for (std::size_t index = 0u; index < num_events; ++index)
        std::cout << index << ' ' << integer_to_bits_string(state_ptr->generated_events()[index], state_ptr->total_num_qubits()) << '\n';
      std::cout
        << "Events finished: "
        << duration_to_second(start_time, finish_time_and_process.first)
        << " ("
        << duration_to_second(last_processed_time, finish_time_and_process.first)
        << ')'
        << std::endl;
      break;
    }
  }

  return EXIT_SUCCESS;
}


#undef BRA_clock
#undef BRA_chrono
#undef BRA_mt19937_64

