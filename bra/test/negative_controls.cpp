// Compile with BRA_NO_MPI and link bra objects except bra.o.
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <bra/control_io.hpp>
#include <bra/state.hpp>
#include <bra/gate/controlled_hadamard.hpp>
#include <bra/gate/multi_controlled_u1.hpp>
#include <bra/gate/toffoli.hpp>

namespace
{
  auto require(bool const condition) -> void
  {
    if (not condition)
      throw std::runtime_error{"control value/representation mismatch"};
  }

  auto tokens(std::string const& text) -> std::vector<std::string>
  {
    auto stream = std::istringstream{text};
    auto result = std::vector<std::string>{};
    for (auto token = std::string{}; stream >> token; )
      result.push_back(token);
    return result;
  }
}

int main()
{
  auto const zero = bra::control_qubit_type{bra::qubit_type{3u}, 0u};
  auto const one = bra::control_qubit_type{bra::qubit_type{0u}, 1u};
  require(bra::control_operand(zero) == "3#0");
  require(bra::control_operand(one) == "0");
  bra::gate::controlled_hadamard negative{bra::qubit_type{1u}, zero};
  bra::gate::controlled_hadamard positive{bra::qubit_type{1u}, one};
  bra::gate::toffoli mixed{bra::qubit_type{1u}, zero, one};
  bra::gate::multi_controlled_u1 phase{0.5, {zero, one, {bra::qubit_type{2u}, 0u}}};
  require(tokens(negative.representation()) == std::vector<std::string>{"CH", "3#0", "1"});
  require(tokens(positive.representation()) == std::vector<std::string>{"CH", "0", "1"});
  require(tokens(mixed.representation()) == std::vector<std::string>{"Toffoli", "3#0", "0", "1"});
  require(tokens(phase.representation()) == std::vector<std::string>{"CCU1", "3#0", "0", "2#0", "0.5"});

  // Negative controls cannot be factored into the positive-only outer mask.
  auto found = std::vector<bra::found_qubit>(4u, bra::found_qubit::not_found);
  bra::set_found_qubits(found, one);
  bra::set_found_qubits(found, zero);
  require(found[0] == bra::found_qubit::control_qubit);
  require(found[3] == bra::found_qubit::qubit);
  bra::set_found_qubits(found, bra::control_qubit_type{bra::qubit_type{0u}, 0u});
  bra::set_found_qubits(found, one);
  require(found[0] == bra::found_qubit::qubit);
}
