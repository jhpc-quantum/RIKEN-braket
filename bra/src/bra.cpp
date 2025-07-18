#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <random>
#include <chrono>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cxxopts.hpp>

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
# include <bra/make_simple_mpi_state.hpp>
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
  auto communicator = yampi::communicator{yampi::tags::world_communicator};
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
  auto options = cxxopts::Options{"bra", "Massively parallel full-state simulator of quantum circuits"};
  options.add_options()
    ("m,mode", "set mode, \"simple\" or \"unit\"", cxxopts::value<std::string>()->default_value("simple"))
    ("f,file", "set the name of input qcx file, or read from standard input if this option is unspecified", cxxopts::value<std::string>())
#ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    ("buffer-size", "set the number of complex numbers in buffer (meaningful only if the value of page-qubits is 0)", cxxopts::value<unsigned int>()->default_value("65536"))
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
    ("unit-qubits", "set the number of unit qubits (meaningful only for unit mode)", cxxopts::value<unsigned int>())
    ("unit-processes", "set the number of MPI processes for each unit (meaningful only for unit mode)", cxxopts::value<unsigned int>())
    ("threads", "set the number of threads per process", cxxopts::value<unsigned int>()->default_value("1"))
    ("page-qubits", "set the number of page qubits", cxxopts::value<unsigned int>()->default_value("2"))
    ("seed", "set seed of random number generator", cxxopts::value<seed_type>()->default_value("1"))
    ("h,help", "print this information")
    ;
#else // BRA_NO_MPI
  auto options = cxxopts::Options{"bra", "Full-state simulator of quantum circuits (single-process ver.)"};
  options.add_options()
    ("f,file", "set the name of input qcx file, or read from standard input if this option is unspecified", cxxopts::value<std::string>())
    ("threads", "set the number of threads", cxxopts::value<unsigned int>()->default_value("1"))
    ("seed", "set seed of random number generator", cxxopts::value<seed_type>()->default_value("1"))
    ("h,help", "print this information")
    ;
#endif // BRA_NO_MPI

  auto parse_result = options.parse(argc, argv);

  if (parse_result.count("help"))
  {
#ifndef BRA_NO_MPI
    if (is_io_root_rank)
      std::cout << options.help() << std::endl;
#else // BRA_NO_MPI
    std::cout << options.help() << std::endl;
#endif // BRA_NO_MPI
    return EXIT_SUCCESS;
  }

#ifndef BRA_NO_MPI
  auto const mpi_mode = parse_result["mode"].as<std::string>();
  auto const is_simple = mpi_mode == "simple";
  auto const is_unit = mpi_mode == "unit";
  if (is_unit and ((not parse_result.count("unit-qubits")) or (not parse_result.count("unit-processes"))))
  {
    if (is_io_root_rank)
      std::cerr << "Error: wrong number of arguments\n" << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  auto const num_page_qubits = parse_result["page-qubits"].as<unsigned int>();
#ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto const num_elements_in_buffer = parse_result["buffer-size"].as<unsigned int>();

  if (num_elements_in_buffer == 0u)
  {
    if (is_io_root_rank)
      std::cerr << "Error: wrong argument\n" << options.help() << std::flush;
    return EXIT_FAILURE;
  }
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS

  auto num_unit_qubits = 0u;
  auto num_processes_per_unit = 1u;

  if (is_unit)
  {
    num_unit_qubits = parse_result["unit-qubits"].as<unsigned int>();
    num_processes_per_unit = parse_result["unit-processes"].as<unsigned int>();

    if (num_processes_per_unit == 0u)
    {
      if (is_io_root_rank)
        std::cerr << "Error: wrong argument\n" << options.help() << std::flush;
      return EXIT_FAILURE;
    }
  }
  else if (not is_simple)
  {
    if (is_io_root_rank)
      std::cerr << "Error: wrong argument\n" << options.help() << std::flush;
    return EXIT_FAILURE;
  }
#endif // BRA_NO_MPI

  auto const num_threads_per_process = parse_result["threads"].as<unsigned int>();
  auto const seed = parse_result["seed"].as<seed_type>();

  std::ifstream possible_input_stream;
  if (parse_result.count("file"))
  {
    auto const filename = parse_result["file"].as<std::string>();
    if (not filename.empty())
    {
      possible_input_stream.open(filename);
      if (not possible_input_stream)
      {
#ifndef BRA_NO_MPI
        if (is_io_root_rank)
          std::cerr << "ERROR: cannot open an input file " << filename << '\n' << options.help() << std::endl;
#else // BRA_NO_MPI
        std::cerr << "ERROR: cannot open an input file " << filename << '\n' << options.help() << std::endl;
#endif // BRA_NO_MPI
        return EXIT_FAILURE;
      }
    }
  }


#ifndef BRA_NO_MPI
  auto gates = bra::gates{parse_result.count("file") ? possible_input_stream : std::cin, num_unit_qubits, num_processes_per_unit, environment, root_rank, communicator};
# ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto state_ptr
    = is_unit
      ? bra::make_unit_mpi_state(
          num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), num_unit_qubits, gates.initial_permutation(),
          num_threads_per_process, num_processes_per_unit, seed, communicator, environment)
      : bra::make_simple_mpi_state(
          num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), gates.initial_permutation(),
          num_threads_per_process, seed, communicator, environment);
# else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto state_ptr
    = is_unit
      ? bra::make_unit_mpi_state(
          num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), num_unit_qubits, gates.initial_permutation(),
          num_threads_per_process, num_processes_per_unit, seed, num_elements_in_buffer, communicator, environment)
      : bra::make_simple_mpi_state(
          num_page_qubits, gates.initial_state_value(), gates.num_lqubits(), gates.initial_permutation(),
          num_threads_per_process, seed, num_elements_in_buffer, communicator, environment);
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
#else // BRA_NO_MPI
  auto gates = bra::gates{parse_result.count("file") ? possible_input_stream : std::cin};
  auto state_ptr
    = bra::make_nompi_state(gates.initial_state_value(), gates.num_qubits(), num_threads_per_process, seed);
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
        << fmt::format("Expectation values of spins:\n{:^8s} {:^8s} {:^8s}\n", "<Qx>", "<Qy>", "<Qz>");
      for (auto const& spin: *(state_ptr->maybe_expectation_values()))
        std::cout
          << fmt::format(
               "{:^ 8.3f} {:^ 8.3f} {:^ 8.3f}\n",
               0.5 - static_cast<double>(spin[0u]), 0.5 - static_cast<double>(spin[1u]), 0.5 - static_cast<double>(spin[2u]));
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
}


#undef BRA_clock
