#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <algorithm>
#include <utility>
#include <random>
#include <chrono>
#include <memory>

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

#include <bra/interpreter.hpp>
#include <bra/state.hpp>
#ifndef BRA_NO_MPI
# include <bra/make_simple_mpi_state.hpp>
# include <bra/make_unit_mpi_state.hpp>
#else
# include <bra/nompi_state.hpp>
#endif


int main(int argc, char* argv[])
{
  std::ios::sync_with_stdio(false);

  using rng_type = std::mt19937_64;
  using seed_type = rng_type::result_type;

#ifndef BRA_NO_MPI
  yampi::environment environment{argc, argv, yampi::thread_support::funneled};
  auto const world_communicator = yampi::communicator{yampi::tags::world_communicator};
  auto const world_rank = world_communicator.rank(environment);
  auto const num_processes = world_communicator.size(environment);
  using namespace yampi::literals::rank_literals;
  auto const is_io_root_rank = world_rank == 0_r and yampi::is_io_process(0_r, environment);

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
      std::cerr << "Error: unit-qubits and unit-processes should be specified\n" << options.help() << std::endl;
    return EXIT_FAILURE;
  }

  auto const num_page_qubits = parse_result["page-qubits"].as<unsigned int>();
#ifdef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto const num_elements_in_buffer = parse_result["buffer-size"].as<unsigned int>();

  if (num_elements_in_buffer == 0u)
  {
    if (is_io_root_rank)
      std::cerr << "Error: buffer-size should be greater than 0\n" << options.help() << std::flush;
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
        std::cerr << "Error: unit-processes should be greater than 0\n" << options.help() << std::flush;
      return EXIT_FAILURE;
    }
  }
  else if (not is_simple)
  {
    if (is_io_root_rank)
      std::cerr << "Error: mode should be specified as \"simple\" or \"unit\"\n" << options.help() << std::flush;
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
  auto interpreter = bra::interpreter{parse_result.count("file") ? possible_input_stream : std::cin, num_unit_qubits, num_processes_per_unit, environment, 0_r, world_communicator};
  if (interpreter.largest_num_operated_qubits() > interpreter.num_lqubits() - num_page_qubits)
  {
    if (is_io_root_rank)
      std::cerr << "Error: the largest number of operated qubits " << interpreter.largest_num_operated_qubits() << " should be less than the number of non-page qubits " << (interpreter.num_lqubits() - num_page_qubits) << '\n' << options.help() << std::flush;
    return EXIT_FAILURE;
  }

  auto const num_circuits = interpreter.num_circuits();
  if (num_processes % num_circuits != 0u)
  {
    if (is_io_root_rank)
      std::cerr << "Error: the number of MPI processes should be proportional to the number of simulated quantum circuits\n" << options.help() << std::flush;
    return EXIT_FAILURE;
  }
  auto const num_processes_per_circuit = num_processes / num_circuits;

  auto const circuit_index = static_cast<int>(world_rank) / static_cast<int>(num_processes_per_circuit);
  auto const intercircuit_index = static_cast<int>(world_rank) % static_cast<int>(num_processes_per_circuit);
  auto const circuit_communicator
    = yampi::communicator{world_communicator, yampi::color{circuit_index}, intercircuit_index, environment};
  auto const intercircuit_communicator
    = yampi::communicator{world_communicator, yampi::color{intercircuit_index}, circuit_index, environment};

  auto intercommunicators = std::vector<yampi::intercommunicator>{};
  intercommunicators.reserve(num_circuits - decltype(num_circuits){1});
  for (auto remote_circuit_index = 0; remote_circuit_index < static_cast<int>(num_circuits); ++remote_circuit_index)
  {
    if (remote_circuit_index == circuit_index)
      continue;

    auto const remote_leader = yampi::rank{static_cast<int>(num_processes_per_circuit) * remote_circuit_index};
    auto const tag
      = circuit_index > remote_circuit_index
        ? yampi::tag{circuit_index * static_cast<int>(num_circuits) + remote_circuit_index}
        : yampi::tag{remote_circuit_index * static_cast<int>(num_circuits) + circuit_index};
    intercommunicators.emplace_back(circuit_communicator, 0_r, world_communicator, remote_leader, tag, environment);
  }

# ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto state_ptr
    = is_unit
      ? bra::make_unit_mpi_state(
          num_page_qubits, interpreter.initial_state_value(), interpreter.num_lqubits(), num_unit_qubits, interpreter.initial_permutation(),
          num_threads_per_process, num_processes_per_unit, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment)
      : bra::make_simple_mpi_state(
          num_page_qubits, interpreter.initial_state_value(), interpreter.num_lqubits(), interpreter.initial_permutation(),
          num_threads_per_process, seed, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment);
# else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
  auto state_ptr
    = is_unit
      ? bra::make_unit_mpi_state(
          num_page_qubits, interpreter.initial_state_value(), interpreter.num_lqubits(), num_unit_qubits, interpreter.initial_permutation(),
          num_threads_per_process, num_processes_per_unit, seed, num_elements_in_buffer, circuit_communicator, circuit_index, intercircuit_communicator, intercommunicators, environment)
      : bra::make_simple_mpi_state(
          num_page_qubits, interpreter.initial_state_value(), interpreter.num_lqubits(), interpreter.initial_permutation(),
          num_threads_per_process, seed, num_elements_in_buffer, circuit_communicator, intercircuit_communicator, circuit_index, intercommunicators, environment);
# endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS

  interpreter.apply_circuit(*state_ptr, circuit_index);

  if (not is_io_root_rank)
    return EXIT_SUCCESS;
#else // BRA_NO_MPI
  auto interpreter = bra::interpreter{parse_result.count("file") ? possible_input_stream : std::cin};
  if (interpreter.largest_num_operated_qubits() > interpreter.num_qubits())
  {
    std::cerr << "Error: the largest number of operated qubits " << interpreter.largest_num_operated_qubits() << " should be less than the number of qubits " << interpreter.num_qubits() << '\n' << options.help() << std::flush;
    return EXIT_FAILURE;
  }

  auto const num_circuits = interpreter.num_circuits();

  auto nompi_states = std::vector< ::bra::nompi_state >{};
  nompi_states.reserve(num_circuits);
  for (auto circuit_index = 0; circuit_index < static_cast<int>(num_circuits); ++circuit_index)
    nompi_states.emplace_back(interpreter.initial_state_value(), interpreter.num_qubits(), num_threads_per_process, seed, circuit_index);

  while (true)
  {
    for (auto circuit_index = 0; circuit_index < static_cast<int>(num_circuits); ++circuit_index)
      if (not nompi_states[circuit_index].is_waiting())
        interpreter.apply_circuit(nompi_states[circuit_index], circuit_index);

    using std::begin;
    using std::end;
    if (std::none_of(begin(nompi_states), end(nompi_states), [](::bra::nompi_state const& state) { return state.is_waiting(); }))
      break;

    auto is_inner_product_all = true;
    auto is_inner_product_all_op = true;
    for (auto circuit_index = 0; circuit_index < static_cast<int>(num_circuits); ++circuit_index)
    {
      if (not nompi_states[circuit_index].is_waiting())
      {
        is_inner_product_all = false;
        is_inner_product_all_op = false;
        continue;
      }

      if (nompi_states[circuit_index].wait_reason().is_inner_product_all())
      {
        is_inner_product_all_op = false;
        continue;
      }

      if (nompi_states[circuit_index].wait_reason().is_inner_product_all_op())
      {
        is_inner_product_all = false;
        continue;
      }

      is_inner_product_all = false;
      is_inner_product_all_op = false;

      if (nompi_states[circuit_index].wait_reason().is_inner_product())
      {
        auto const other_circuit_index = nompi_states[circuit_index].wait_reason().other_circuit_index();
        if (not nompi_states[other_circuit_index].is_waiting())
          continue;

        if (nompi_states[other_circuit_index].wait_reason().is_inner_product() and nompi_states[other_circuit_index].wait_reason().other_circuit_index() == circuit_index)
        {
          ::bra::inner_product(nompi_states[circuit_index], nompi_states[other_circuit_index]);

          nompi_states[circuit_index].cancel_waiting();
          nompi_states[other_circuit_index].cancel_waiting();

          continue;
        }
      }
      else if (nompi_states[circuit_index].wait_reason().is_inner_product_op())
      {
        auto const other_circuit_index = nompi_states[circuit_index].wait_reason().other_circuit_index();
        if (not nompi_states[other_circuit_index].is_waiting())
          continue;

        if (nompi_states[other_circuit_index].wait_reason().is_inner_product_op() and nompi_states[other_circuit_index].wait_reason().other_circuit_index() == circuit_index)
        {
          ::bra::inner_product_op(
            nompi_states[circuit_index], nompi_states[other_circuit_index],
            nompi_states[circuit_index].wait_reason().operator_literal_or_variable_name(),
            nompi_states[circuit_index].wait_reason().operated_qubits());

          nompi_states[circuit_index].cancel_waiting();
          nompi_states[other_circuit_index].cancel_waiting();

          continue;
        }
      }
    }

    if (is_inner_product_all)
    {
      ::bra::inner_product_all(nompi_states);

      for (auto& state: nompi_states)
        state.cancel_waiting();

      continue;
    }
    else if (is_inner_product_all_op)
    {
      ::bra::inner_product_all_op(
        nompi_states,
        nompi_states.front().wait_reason().operator_literal_or_variable_name(),
        nompi_states.front().wait_reason().operated_qubits());

      for (auto& state: nompi_states)
        state.cancel_waiting();

      continue;
    }
  }
#endif // BRA_NO_MPI
}

