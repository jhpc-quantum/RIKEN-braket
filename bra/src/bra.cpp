#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <random>
#include <chrono>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

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
# include <bra/make_unit_mpi_state.hpp>
#else
# include <bra/nompi_state.hpp>
#endif

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else // BRA_NO_MPI
# define BRA_clock std::chrono::system_clock
#endif // BRA_NO_MPI


template <typename StateInteger, typename BitInteger>
std::string integer_to_bits_string(StateInteger const integer, BitInteger const total_num_qubits)
{
  auto result = std::string{};
  result.reserve(total_num_qubits);

  for (auto left_bit = total_num_qubits; left_bit > BitInteger{0u}; --left_bit)
  {
    auto const zero_or_one
      = (integer bitand (StateInteger{1u} << (left_bit - BitInteger{1u})))
        >> (left_bit - BitInteger{1u});
    if (zero_or_one == StateInteger{0u})
      result.push_back('0');
    else
      result.push_back('1');
  }
  return result;
}

template <typename Clock, typename Duration>
double duration_to_second(
  std::chrono::time_point<Clock, Duration> const& from,
  std::chrono::time_point<Clock, Duration> const& to)
{ return 0.000001 * std::chrono::duration_cast<std::chrono::microseconds>(to - from).count(); }

int main(int argc, char* argv[])
{
  std::ios::sync_with_stdio(false);

  using rng_type = std::mt19937_64;
  using seed_type = rng_type::result_type;

#ifndef BRA_NO_MPI
  yampi::environment environment{argc, argv, yampi::thread_support::funneled};
  auto communicator = yampi::communicator{yampi::tags::world_communicator()};
  auto const rank = communicator.rank(environment);
  constexpr auto root_rank = yampi::rank{0};
  auto const is_io_root_rank = rank == root_rank and yampi::is_io_process(root_rank, environment);

  if (environment.thread_support() == yampi::thread_support::single)
  {
    if (is_io_root_rank)
      std::cerr << "multithread environment is required" << std::endl;
    return EXIT_FAILURE;
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  if (argc < 3 or argc > 7)
  {
    if (is_io_root_rank)
      std::cerr
        << "wrong number of arguments: bra general <qcxfile> [<num_threads_per_process> [<num_page_qubits> [<seed>]]]\n"
           "                           bra unit <qcxfile> <num_unit_qubits> <num_processes_per_unit> [<num_threads_per_process> [<seed>]]\n"
           "  default values are: num_threads_per_process=1, num_page_qubits=2, seed=1\n"
        << std::flush;
    return EXIT_FAILURE;
  }
#else // BRA_NO_MPI
  if (argc < 2 or argc > 4)
  {
    std::cerr
      << "wrong number of arguments: bra qcxfile [num_threads_per_process [seed]]\n"
         "  default values are: num_threads_per_process=1, seed=1\n"
      << std::flush;
    return EXIT_FAILURE;
  }
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  auto const mpi_policy_string = std::string{argv[1]};
  auto const filename = std::string{argv[2]};

  auto num_threads_per_process = 1u;
  auto num_page_qubits = 2u;
  auto seed = seed_type{1u};

  auto num_unit_qubits = 0u;
  auto num_processes_per_unit = 1u;

  if (mpi_policy_string == "general")
  {
    if (argc > 6)
    {
      if (is_io_root_rank)
        std::cerr
          << "wrong number of arguments: bra general <qcxfile> [<num_threads_per_process> [<num_page_qubits> [<seed>]]]\n"
             "                           bra unit <qcxfile> <num_unit_qubits> <num_processes_per_unit> [<num_threads_per_process> [<seed>]]\n"
             "  default values are: num_threads_per_process=1, num_page_qubits=2, seed=1\n"
          << std::flush;
      return EXIT_FAILURE;
    }

    num_threads_per_process
      = argc >= 4
        ? boost::lexical_cast<unsigned int>(argv[3])
        : 1u;
    num_page_qubits
      = argc >= 5
        ? boost::lexical_cast<unsigned int>(argv[4])
        : 2u;
    seed
      = argc == 6
        ? boost::lexical_cast<seed_type>(argv[5])
        : seed_type{1};
  }
  else if (mpi_policy_string == "unit")
  {
    if (argc < 5)
    {
      if (is_io_root_rank)
        std::cerr
          << "wrong number of arguments: bra general <qcxfile> [<num_threads_per_process> [<num_page_qubits> [<seed>]]]\n"
             "                           bra unit <qcxfile> <num_unit_qubits> <num_processes_per_unit> [<num_threads_per_process> [<seed>]]\n"
             "  default values are: num_threads_per_process=1, num_page_qubits=2, seed=1\n"
          << std::flush;
      return EXIT_FAILURE;
    }

    num_unit_qubits = boost::lexical_cast<unsigned int>(argv[3]);
    num_processes_per_unit = boost::lexical_cast<unsigned int>(argv[4]);
    if (num_processes_per_unit == 0u)
    {
      if (is_io_root_rank)
        std::cerr
          << "wrong argument: bra general <qcxfile> [<num_threads_per_process> [<num_page_qubits> [<seed>]]]\n"
             "                bra unit <qcxfile> <num_unit_qubits> <num_processes_per_unit> [<num_threads_per_process> [<seed>]]\n"
             "  default values are: num_thread_per_process=1, num_page_qubits=2, seed=1\n"
          << std::flush;
      return EXIT_FAILURE;
    }

    num_threads_per_process
      = argc >= 6
        ? boost::lexical_cast<unsigned int>(argv[5])
        : 1u;
    seed
      = argc == 7
        ? boost::lexical_cast<seed_type>(argv[6])
        : seed_type{1};
  }
  else
  {
    if (is_io_root_rank)
      std::cerr
        << "wrong argument: bra general <qcxfile> [<num_threads_per_process> [<num_page_qubits> [<seed>]]]\n"
           "                bra unit <qcxfile> <num_unit_qubits> <num_processes_per_unit> [<num_threads_per_process> [<seed>]]\n"
           "  default values are: num_thread_per_process=1, num_page_qubits=2, seed=1\n"
        << std::flush;
    return EXIT_FAILURE;
  }
#else // BRA_NO_MPI
  auto const filename = std::string{argv[1]};
  auto const num_threads
    = argc >= 3
      ? boost::lexical_cast<unsigned int>(argv[2])
      : 1u;
  auto const seed
    = argc == 4
      ? boost::lexical_cast<seed_type>(argv[3])
      : seed_type{1};
#endif // BRA_NO_MPI

  auto file_stream = std::ifstream{filename.c_str()};
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
  auto gates = bra::gates{file_stream, num_unit_qubits, num_processes_per_unit, environment, root_rank, communicator};
  auto state_ptr
    = mpi_policy_string == "unit"
      ? bra::make_unit_mpi_state(
          gates.initial_state_value(), gates.num_lqubits(), num_unit_qubits, gates.initial_permutation(),
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment)
      : bra::make_general_mpi_state(
          num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), gates.initial_permutation(),
          num_threads_per_process, seed, communicator, environment);
#else // BRA_NO_MPI
  auto gates = bra::gates{file_stream};
  auto state_ptr
    = bra::make_nompi_state(gates.initial_state_value(), gates.num_qubits(), num_threads, seed);
#endif // BRA_NO_MPI

#ifndef BRA_NO_MPI
  auto const start_time = BRA_clock::now(environment);
#else
  auto const start_time = BRA_clock::now();
#endif
  auto last_processed_time = start_time;

  *state_ptr << gates;

#ifndef BRA_NO_MPI
  if (not is_io_root_rank)
    return EXIT_SUCCESS;
#endif

  auto const num_finish_processes = state_ptr->num_finish_processes();
  for (auto index = decltype(num_finish_processes){0u}; index < num_finish_processes; ++index)
  {
    auto finish_time_and_process = state_ptr->finish_time_and_process(index);

    if (finish_time_and_process.second == ::bra::finished_process::operations)
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
    else if (finish_time_and_process.second == ::bra::finished_process::begin_measurement)
    {
      std::cout
        << boost::format("Expectation values of spins:\n%|=8s| %|=8s| %|=8s|\n")
           % "<Qx>" % "<Qy>" % "<Qz>";
      for (auto const& spin: *(state_ptr->maybe_expectation_values()))
        std::cout
          << boost::format("%|=8.3f| %|=8.3f| %|=8.3f|\n")
             % (0.5-spin[0u]) % (0.5-spin[1u]) % (0.5-spin[2u]);
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
    else if (finish_time_and_process.second == ::bra::finished_process::ket_measure)
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
    else if (finish_time_and_process.second == ::bra::finished_process::generate_events)
    {
      std::cout << "Events:\n";
      auto const num_events = state_ptr->generated_events().size();
      for (auto index = decltype(num_events){0u}; index < num_events; ++index)
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
