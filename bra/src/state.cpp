#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <utility>
#ifdef BRA_NO_MPI
# include <chrono>
# include <memory>
#endif
#include <stdexcept>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <boost/variant/variant.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#ifndef BRA_NO_MPI
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/wall_clock.hpp>
#endif // BRA_NO_MPI

#include <ket/qubit.hpp>
#include <ket/control.hpp>
#include <ket/utility/imaginary_unit.hpp>

#include <bra/state.hpp>
#include <bra/utility/closest_floating_point_of.hpp>

#ifndef BRA_NO_MPI
# define BRA_clock yampi::wall_clock
#else
# define BRA_clock std::chrono::system_clock
#endif


namespace bra
{
  too_many_operated_qubits_error::too_many_operated_qubits_error(std::size_t const num_operated_qubits, std::size_t const max_num_operated_qubits)
    : std::runtime_error{std::string{"the number of operated qubits ("}.append(std::to_string(num_operated_qubits)).append(") is larger than its maximum value (").append(std::to_string(max_num_operated_qubits)).append(")").c_str()}
  { }

  unsupported_fused_gate_error::unsupported_fused_gate_error(std::string const& mnemonic)
    : std::runtime_error{(mnemonic + " is not supported in gate fusion").c_str()}
  { }

  wrong_assignment_argument_error::wrong_assignment_argument_error(std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name)
    : std::runtime_error{(std::string{"\""} + lhs_variable_name + " " + to_string(op) + " " + rhs_literal_or_variable_name + "\" is a wrong argument").c_str()}
  { }

  std::string wrong_assignment_argument_error::to_string(::bra::assign_operation_type const op)
  {
    return
      op == ::bra::assign_operation_type::assign
      ? ":="
      : op == ::bra::assign_operation_type::plus_assign
        ? "+="
        : op == ::bra::assign_operation_type::minus_assign
          ? "-="
          : op == ::bra::assign_operation_type::multiplies_assign
            ? "*="
            : op == ::bra::assign_operation_type::divides_assign
              ? "/="
              : "";
  }

  wrong_comparison_argument_error::wrong_comparison_argument_error(std::string const& lhs_variable_name, ::bra::compare_operation_type const op, std::string const& rhs_literal_or_variable_name)
    : std::runtime_error{(std::string{"\""} + lhs_variable_name + " " + to_string(op) + " " + rhs_literal_or_variable_name + "\" is a wrong argument").c_str()}
  { }

  std::string wrong_comparison_argument_error::to_string(::bra::compare_operation_type const op)
  {
    return
      op == ::bra::compare_operation_type::equal_to
      ? "=="
      : op == ::bra::compare_operation_type::not_equal_to
        ? "\\="
        : op == ::bra::compare_operation_type::greater
          ? ">"
          : op == ::bra::compare_operation_type::less
            ? "<"
            : op == ::bra::compare_operation_type::greater_equal
              ? ">="
              : op == ::bra::compare_operation_type::less_equal
                ? "<="
                : "";
  }

  wrong_pauli_string_length_error::wrong_pauli_string_length_error(std::size_t const num_operated_qubits, std::size_t const pauli_string_length)
    : std::runtime_error{std::string{"the number of operated qubits ("}.append(std::to_string(num_operated_qubits)).append(") is not equal to the length of Pauli string (").append(std::to_string(pauli_string_length)).append(")").c_str()}
  { }

#ifndef BRA_NO_MPI
  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      last_measured_qubit_{qubit_type{total_num_qubits_}},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      result_{},
      is_in_fusion_{false},
      found_qubits_{},
      random_number_generator_{seed},
      permutation_{static_cast<permutation_type::size_type>(total_num_qubits)},
      buffer_{},
      communicator_{communicator},
      environment_{environment},
      start_time_{BRA_clock::now(environment_)},
      last_processed_time_{start_time_},
      phase_coefficients_{},
      maybe_label_{},
      real_variables_{},
      complex_variables_{},
      int_variables_{},
      pauli_string_space_variables_{}
  {
    found_qubits_.reserve(total_num_qubits_);
    ket::utility::generate_phase_coefficients(phase_coefficients_, total_num_qubits_);
  }

  state::state(
    bit_integer_type const total_num_qubits,
    seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      last_measured_qubit_{qubit_type{total_num_qubits_}},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      result_{},
      is_in_fusion_{false},
      found_qubits_{},
      random_number_generator_{seed},
      permutation_{static_cast<permutation_type::size_type>(total_num_qubits)},
      buffer_(num_elements_in_buffer),
      communicator_{communicator},
      environment_{environment},
      start_time_{BRA_clock::now(environment_)},
      last_processed_time_{start_time_},
      phase_coefficients_{},
      maybe_label_{},
      real_variables_{},
      complex_variables_{},
      int_variables_{},
      pauli_string_space_variables_{}
  {
    found_qubits_.reserve(total_num_qubits_);
    ket::utility::generate_phase_coefficients(phase_coefficients_, total_num_qubits_);
  }

  state::state(
    std::vector<permutated_qubit_type> const& initial_permutation,
    seed_type const seed,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{static_cast<bit_integer_type>(initial_permutation.size())},
      last_outcomes_{total_num_qubits_, ket::gate::outcome::unspecified},
      last_measured_qubit_{qubit_type{total_num_qubits_}},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      result_{},
      is_in_fusion_{false},
      found_qubits_{},
      random_number_generator_{seed},
      permutation_{
        std::begin(initial_permutation), std::end(initial_permutation)},
      buffer_{},
      communicator_{communicator},
      environment_{environment},
      start_time_{BRA_clock::now(environment_)},
      last_processed_time_{start_time_},
      phase_coefficients_{},
      maybe_label_{},
      real_variables_{},
      complex_variables_{},
      int_variables_{},
      pauli_string_space_variables_{}
  {
    found_qubits_.reserve(total_num_qubits_);
    ket::utility::generate_phase_coefficients(phase_coefficients_, total_num_qubits_);
  }

  state::state(
    std::vector<permutated_qubit_type> const& initial_permutation,
    seed_type const seed,
    unsigned int const num_elements_in_buffer,
    yampi::communicator const& communicator,
    yampi::environment const& environment)
    : total_num_qubits_{static_cast<bit_integer_type>(initial_permutation.size())},
      last_outcomes_{total_num_qubits_, ket::gate::outcome::unspecified},
      last_measured_qubit_{qubit_type{total_num_qubits_}},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      result_{},
      is_in_fusion_{false},
      found_qubits_{},
      random_number_generator_{seed},
      permutation_{
        std::begin(initial_permutation), std::end(initial_permutation)},
      buffer_(num_elements_in_buffer),
      communicator_{communicator},
      environment_{environment},
      start_time_{BRA_clock::now(environment_)},
      last_processed_time_{start_time_},
      phase_coefficients_{},
      maybe_label_{},
      real_variables_{},
      complex_variables_{},
      int_variables_{},
      pauli_string_space_variables_{}
  {
    found_qubits_.reserve(total_num_qubits_);
    ket::utility::generate_phase_coefficients(phase_coefficients_, total_num_qubits_);
  }
#else // BRA_NO_MPI
  state::state(bit_integer_type const total_num_qubits, seed_type const seed)
    : total_num_qubits_{total_num_qubits},
      last_outcomes_{total_num_qubits, ket::gate::outcome::unspecified},
      last_measured_qubit_{qubit_type{total_num_qubits_}},
      maybe_expectation_values_{},
      measured_value_{},
      generated_events_{},
      result_{},
      is_in_fusion_{false},
      found_qubits_{},
      random_number_generator_{seed},
      start_time_{BRA_clock::now()},
      last_processed_time_{start_time_},
      phase_coefficients_{},
      maybe_label_{},
      real_variables_{},
      complex_variables_{},
      int_variables_{},
      pauli_string_space_variables_{}
  {
    found_qubits_.reserve(total_num_qubits_);
    ket::utility::generate_phase_coefficients(phase_coefficients_, total_num_qubits_);
  }
#endif // BRA_NO_MPI

  void state::generate_new_variable(std::string const& variable_name, ::bra::variable_type const type, int const num_elements)
  {
    if (type == ::bra::variable_type::real)
      generate_new_real_variable(variable_name, num_elements);
    else if (type == ::bra::variable_type::complex_)
      generate_new_complex_variable(variable_name, num_elements);
    else if (type == ::bra::variable_type::integer)
      generate_new_int_variable(variable_name, num_elements);
    else if (type == ::bra::variable_type::pauli_string_space)
      generate_new_pauli_string_space_variable(variable_name, num_elements);
  }

  void state::generate_new_real_variable(std::string const& variable_name, int const num_elements)
  {
    using std::end;
    if (real_variables_.find(variable_name) != end(real_variables_))
      throw 1;

    if (num_elements <= 0)
      throw 1;

    using size_type = std::vector<real_type>::size_type;
    real_variables_[variable_name] = std::vector<real_type>(static_cast<size_type>(num_elements));
  }

  void state::generate_new_complex_variable(std::string const& variable_name, int const num_elements)
  {
    using std::end;
    if (complex_variables_.find(variable_name) != end(complex_variables_))
      throw 1;

    if (num_elements <= 0)
      throw 1;

    using size_type = std::vector<complex_type>::size_type;
    complex_variables_[variable_name] = std::vector<complex_type>(static_cast<size_type>(num_elements));
  }

  void state::generate_new_int_variable(std::string const& variable_name, int const num_elements)
  {
    using std::end;
    if (int_variables_.find(variable_name) != end(int_variables_))
      throw 1;

    if (num_elements <= 0)
      throw 1;

    using size_type = std::vector<int_type>::size_type;
    int_variables_[variable_name] = std::vector<int_type>(static_cast<size_type>(num_elements));
  }

  void state::generate_new_pauli_string_space_variable(std::string const& variable_name, int const num_elements)
  {
    using std::end;
    if (pauli_string_space_variables_.find(variable_name) != end(pauli_string_space_variables_))
      throw 1;

    if (num_elements <= 0)
      throw 1;

    using size_type = std::vector<std::unordered_map<std::string, complex_type>>::size_type;
    pauli_string_space_variables_[variable_name] = std::vector< ::bra::pauli_string_space >(static_cast<size_type>(num_elements));
  }

  void state::invoke_assign_operation(
    std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name)
  {
    if (not std::isalpha(static_cast<unsigned char>(lhs_variable_name.front())))
      throw ::bra::wrong_assignment_argument_error{lhs_variable_name, op, rhs_literal_or_variable_name};

    using size_type = std::string::size_type;
    auto const found_index = lhs_variable_name.find(':');
    auto const variable_name = lhs_variable_name.substr(size_type{0u}, found_index);
    auto const index = found_index == std::string::npos ? 0 : to_int(lhs_variable_name.substr(found_index + size_type{1u}));

    if (real_variables_.find(variable_name) != end(real_variables_))
    {
      auto const rhs_value = to_real(rhs_literal_or_variable_name);
      if (op == ::bra::assign_operation_type::assign)
        real_variables_.at(variable_name)[index] = rhs_value;
      else if (op == ::bra::assign_operation_type::plus_assign)
        real_variables_.at(variable_name)[index] += rhs_value;
      else if (op == ::bra::assign_operation_type::minus_assign)
        real_variables_.at(variable_name)[index] -= rhs_value;
      else if (op == ::bra::assign_operation_type::multiplies_assign)
        real_variables_.at(variable_name)[index] *= rhs_value;
      else if (op == ::bra::assign_operation_type::divides_assign)
        real_variables_.at(variable_name)[index] /= rhs_value;
    }
    else if (complex_variables_.find(variable_name) != end(complex_variables_))
    {
      auto const rhs_value = to_complex(rhs_literal_or_variable_name);
      if (op == ::bra::assign_operation_type::assign)
        complex_variables_.at(variable_name)[index] = rhs_value;
      else if (op == ::bra::assign_operation_type::plus_assign)
        complex_variables_.at(variable_name)[index] += rhs_value;
      else if (op == ::bra::assign_operation_type::minus_assign)
        complex_variables_.at(variable_name)[index] -= rhs_value;
      else if (op == ::bra::assign_operation_type::multiplies_assign)
        complex_variables_.at(variable_name)[index] *= rhs_value;
      else if (op == ::bra::assign_operation_type::divides_assign)
        complex_variables_.at(variable_name)[index] /= rhs_value;
    }
    else if (int_variables_.find(variable_name) != end(int_variables_))
    {
      auto const rhs_value = to_int(rhs_literal_or_variable_name);
      if (op == ::bra::assign_operation_type::assign)
        int_variables_.at(variable_name)[index] = rhs_value;
      else if (op == ::bra::assign_operation_type::plus_assign)
        int_variables_.at(variable_name)[index] += rhs_value;
      else if (op == ::bra::assign_operation_type::minus_assign)
        int_variables_.at(variable_name)[index] -= rhs_value;
      else if (op == ::bra::assign_operation_type::multiplies_assign)
        int_variables_.at(variable_name)[index] *= rhs_value;
      else if (op == ::bra::assign_operation_type::divides_assign)
        int_variables_.at(variable_name)[index] /= rhs_value;
    }
    else if (pauli_string_space_variables_.find(variable_name) != end(pauli_string_space_variables_))
    {
      auto& lhs_pauli_string_space = pauli_string_space_variables_.at(variable_name)[index];

      if (op == ::bra::assign_operation_type::assign)
        lhs_pauli_string_space = to_pauli_string_space(rhs_literal_or_variable_name);
      else if (op == ::bra::assign_operation_type::plus_assign)
        for (auto const& rhs_basis_scalar: to_pauli_string_space(rhs_literal_or_variable_name))
          lhs_pauli_string_space[rhs_basis_scalar.first] += rhs_basis_scalar.second;
      else if (op == ::bra::assign_operation_type::minus_assign)
        for (auto const& rhs_basis_scalar: to_pauli_string_space(rhs_literal_or_variable_name))
          lhs_pauli_string_space[rhs_basis_scalar.first] -= rhs_basis_scalar.second;
      else if (op == ::bra::assign_operation_type::multiplies_assign)
      {
        auto const rhs_value = to_complex(rhs_literal_or_variable_name);
        for (auto& pauli_string_scalar: lhs_pauli_string_space)
          pauli_string_scalar.second *= rhs_value;
      }
      else if (op == ::bra::assign_operation_type::divides_assign)
      {
        auto const rhs_value = to_complex(rhs_literal_or_variable_name);
        for (auto& pauli_string_scalar: lhs_pauli_string_space)
          pauli_string_scalar.second /= rhs_value;
      }
    }
    else
      throw ::bra::wrong_assignment_argument_error{lhs_variable_name, op, rhs_literal_or_variable_name};
  }

  void state::generate_print_string(std::ostringstream& oss, std::vector<std::string> const& variables_or_literals)
  {
    bool is_first_argument = true;

    for (auto const& variable_or_literal: variables_or_literals)
    {
      if (is_first_argument)
        is_first_argument = false;
      else
        oss << ' ';

      if (std::isdigit(static_cast<unsigned char>(variable_or_literal.front())) or variable_or_literal.front() == '+' or variable_or_literal.front() == '-' or variable_or_literal.front() == '.')
      {
        oss << variable_or_literal;
        continue;
      }

      if (variable_or_literal.front() == ':')
      {
        using size_type = std::string::size_type;
        auto const variable_name
          = variable_or_literal.substr(size_type{0u}, variable_or_literal.find(':', size_type{1u}));

        if (is_int_symbol(variable_name))
          oss << to_int(variable_or_literal);
        else if (is_real_symbol(variable_name))
          oss << to_real(variable_or_literal);
        else if (is_complex_symbol(variable_name))
          oss << to_complex(variable_or_literal);

        continue;
      }

      using size_type = std::string::size_type;
      auto const variable_name = variable_or_literal.substr(size_type{0u}, variable_or_literal.find(':'));
      using std::end;
      if (int_variables_.find(variable_name) != end(int_variables_))
        oss << to_int(variable_or_literal);
      else if (real_variables_.find(variable_name) != end(real_variables_))
        oss << to_real(variable_or_literal);
      else if (complex_variables_.find(variable_name) != end(complex_variables_))
        oss << to_complex(variable_or_literal);
    }
  }

  void state::invoke_print_operation(std::vector<std::string> const& variables_or_literals)
  {
#ifndef BRA_NO_MPI
    constexpr auto root = yampi::rank{0};
    if (communicator_.rank(environment_) != root)
      return;
#endif // BRA_NO_MPI

    std::ostringstream oss;
    generate_print_string(oss, variables_or_literals);
    std::cout << oss.str() << std::flush;
  }

  void state::invoke_println_operation(std::vector<std::string> const& variables_or_literals)
  {
#ifndef BRA_NO_MPI
    constexpr auto root = yampi::rank{0};
    if (communicator_.rank(environment_) != root)
      return;
#endif // BRA_NO_MPI

    std::ostringstream oss;
    generate_print_string(oss, variables_or_literals);
    oss << '\n';
    std::cout << oss.str() << std::flush;
  }

  void state::invoke_jump_operation(std::string const& label)
  { maybe_label_ = label; }

  void state::invoke_jump_operation(
    std::string const& label,
    std::string const& lhs_variable_name, ::bra::compare_operation_type const op, std::string const& rhs_literal_or_variable_name)
  {
    if (not std::isalpha(static_cast<unsigned char>(lhs_variable_name.front())))
      return;

    using size_type = std::string::size_type;
    auto const found_index = lhs_variable_name.find(':');
    auto const variable_name = lhs_variable_name.substr(size_type{0u}, found_index);
    auto const index = found_index == std::string::npos ? 0 : to_int(lhs_variable_name.substr(found_index + size_type{1u}));

    if (real_variables_.find(variable_name) != end(real_variables_))
    {
      auto const rhs_value = to_real(rhs_literal_or_variable_name);
      if (op == ::bra::compare_operation_type::equal_to
          and real_variables_.at(variable_name)[index] == rhs_value)
        maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::not_equal_to
               and real_variables_.at(variable_name)[index] != rhs_value)
        maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::greater
               and real_variables_.at(variable_name)[index] > rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::less
               and real_variables_.at(variable_name)[index] < rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::greater_equal
               and real_variables_.at(variable_name)[index] >= rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::less_equal
               and real_variables_.at(variable_name)[index] <= rhs_value)
          maybe_label_ = label;
    }
    else if (int_variables_.find(lhs_variable_name) != end(int_variables_))
    {
      auto const rhs_value = to_int(rhs_literal_or_variable_name);
      if (op == ::bra::compare_operation_type::equal_to
          and int_variables_.at(variable_name)[index] == rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::not_equal_to
               and int_variables_.at(variable_name)[index] != rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::greater
               and int_variables_.at(variable_name)[index] > rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::less
               and int_variables_.at(variable_name)[index] < rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::greater_equal
               and int_variables_.at(variable_name)[index] >= rhs_value)
          maybe_label_ = label;
      else if (op == ::bra::compare_operation_type::less_equal
               and int_variables_.at(variable_name)[index] <= rhs_value)
          maybe_label_ = label;
    }
    else
      throw ::bra::wrong_comparison_argument_error{lhs_variable_name, op, rhs_literal_or_variable_name};
  }

  auto state::is_int_symbol(std::string const& symbol_name) const -> bool
  { return symbol_name == ":INT" or symbol_name == ":OUTCOME" or symbol_name == ":OUTCOMES"; }

  auto state::to_int(std::string const& colon_separated_string) const -> int_type
  {
    if (std::isdigit(static_cast<unsigned char>(colon_separated_string.front())) or colon_separated_string.front() == '+' or colon_separated_string.front() == '-')
      return boost::lexical_cast<int_type>(colon_separated_string);

    if (colon_separated_string.front() == ':')
    {
      using std::begin;
      if (colon_separated_string == ":INT")
        return int_type{0};
      else if (colon_separated_string == ":OUTCOME")
        return static_cast<int_type>(static_cast<int>(last_outcomes_[static_cast<bit_integer_type>(last_measured_qubit_)]));

      using size_type = std::string::size_type;
      if (colon_separated_string.size() == size_type{9u} and colon_separated_string == ":OUTCOMES")
        return static_cast<int_type>(static_cast<int>(last_outcomes_.front()));
      else if (colon_separated_string.size() > size_type{10u} and colon_separated_string.substr(size_type{0u}, size_type{10u}) == ":OUTCOMES:")
        return static_cast<int_type>(static_cast<int>(last_outcomes_[to_int(colon_separated_string.substr(size_type{10u}))]));

      constexpr auto int_cast_symbol_length = size_type{5u};
      if (colon_separated_string.size() > int_cast_symbol_length and colon_separated_string.substr(size_type{0u}, int_cast_symbol_length) == ":INT:")
      {
        auto const new_colon_separated_string = colon_separated_string.substr(int_cast_symbol_length);
        if (new_colon_separated_string.empty())
          throw 1;

        if (std::isdigit(static_cast<unsigned char>(new_colon_separated_string.front()))
            or new_colon_separated_string.front() == '+' or new_colon_separated_string.front() == '-'
            or new_colon_separated_string.front() == '.')
          return static_cast<int_type>(boost::lexical_cast<real_type>(new_colon_separated_string));

        auto const variable_name
          = new_colon_separated_string.substr(
              size_type{0u},
              new_colon_separated_string.find(
                ':', new_colon_separated_string.front() == ':' ? size_type{1u} : size_type{0u}));

        if (is_int_symbol(variable_name))
          return to_int(new_colon_separated_string);
        else if (is_real_symbol(variable_name))
          return static_cast<int_type>(to_real(new_colon_separated_string));
        else if (is_complex_symbol(variable_name))
        {
          using std::real;
          return static_cast<int>(real(to_complex(new_colon_separated_string)));
        }

        using std::end;
        if (int_variables_.find(variable_name) != end(int_variables_))
          return to_int(new_colon_separated_string);
        else if (real_variables_.find(variable_name) != end(real_variables_))
          return static_cast<int>(to_real(new_colon_separated_string));
        else if (complex_variables_.find(variable_name) != end(complex_variables_))
        {
          using std::real;
          return static_cast<int>(real(to_complex(new_colon_separated_string)));
        }
      }

      throw 1;
    }

    auto const found_index = colon_separated_string.find(':');
    if (found_index == std::string::npos)
      return int_variables_.at(colon_separated_string).front();

    using size_type = std::string::size_type;
    return int_variables_.at(colon_separated_string.substr(size_type{0u}, found_index))[to_int(colon_separated_string.substr(found_index + size_type{1u}))];
  }

  auto state::is_real_symbol(std::string const& symbol_name) const -> bool
  { return symbol_name == ":REAL" or symbol_name == ":IMAG" or symbol_name == ":PI" or symbol_name == ":HALF_PI" or symbol_name == ":TWO_PI" or symbol_name == ":ROOT_TWO" or symbol_name == ":HALF_ROOT_TWO"; }

  auto state::to_real(std::string const& colon_separated_string) const -> real_type
  {
    if (std::isdigit(static_cast<unsigned char>(colon_separated_string.front())) or colon_separated_string.front() == '+' or colon_separated_string.front() == '-' or colon_separated_string.front() == '.')
      return boost::lexical_cast<real_type>(colon_separated_string);

    if (colon_separated_string.front() == ':')
    {
      if (colon_separated_string == ":REAL")
        return real_type{0};
      else if (colon_separated_string == ":IMAG")
        return real_type{0};
      else if (colon_separated_string == ":PI")
        return boost::math::constants::pi<real_type>();
      else if (colon_separated_string == ":HALF_PI")
        return boost::math::constants::half_pi<real_type>();
      else if (colon_separated_string == ":TWO_PI")
        return boost::math::constants::two_pi<real_type>();
      else if (colon_separated_string == ":ROOT_TWO")
        return boost::math::constants::root_two<real_type>();
      else if (colon_separated_string == ":HALF_ROOT_TWO")
        return boost::math::constants::half_root_two<real_type>();

      using size_type = std::string::size_type;
      constexpr auto real_cast_symbol_length = size_type{6u};
      if (colon_separated_string.size() > real_cast_symbol_length and colon_separated_string.substr(size_type{0u}, real_cast_symbol_length) == ":REAL:")
      {
        auto const new_colon_separated_string = colon_separated_string.substr(real_cast_symbol_length);
        if (new_colon_separated_string.empty())
          throw 1;

        if (std::isdigit(static_cast<unsigned char>(new_colon_separated_string.front()))
            or new_colon_separated_string.front() == '+' or new_colon_separated_string.front() == '-'
            or new_colon_separated_string.front() == '.')
          return boost::lexical_cast<real_type>(new_colon_separated_string);

        auto const variable_name
          = new_colon_separated_string.substr(
              size_type{0u},
              new_colon_separated_string.find(
                ':', new_colon_separated_string.front() == ':' ? size_type{1u} : size_type{0u}));

        if (is_int_symbol(variable_name))
          return static_cast<real_type>(to_int(new_colon_separated_string));
        else if (is_real_symbol(variable_name))
          return to_real(new_colon_separated_string);
        else if (is_complex_symbol(variable_name))
        {
          using std::real;
          return real(to_complex(new_colon_separated_string));
        }

        using std::end;
        if (int_variables_.find(variable_name) != end(int_variables_))
          return static_cast<real_type>(to_int(new_colon_separated_string));
        else if (real_variables_.find(variable_name) != end(real_variables_))
          return to_real(new_colon_separated_string);
        else if (complex_variables_.find(variable_name) != end(complex_variables_))
        {
          using std::real;
          return real(to_complex(new_colon_separated_string));
        }
      }

      constexpr auto imag_cast_symbol_length = size_type{6u};
      if (colon_separated_string.size() > imag_cast_symbol_length and colon_separated_string.substr(size_type{0u}, imag_cast_symbol_length) == ":IMAG:")
      {
        auto const new_colon_separated_string = colon_separated_string.substr(imag_cast_symbol_length);
        if (new_colon_separated_string.empty())
          throw 1;

        if (std::isdigit(static_cast<unsigned char>(new_colon_separated_string.front()))
            or new_colon_separated_string.front() == '+' or new_colon_separated_string.front() == '-'
            or new_colon_separated_string.front() == '.')
          return real_type{0};

        auto const variable_name
          = new_colon_separated_string.substr(
              size_type{0u},
              new_colon_separated_string.find(
                ':', new_colon_separated_string.front() == ':' ? size_type{1u} : size_type{0u}));

        if (is_int_symbol(variable_name))
          return real_type{0};
        else if (is_real_symbol(variable_name))
          return real_type{0};
        else if (is_complex_symbol(variable_name))
        {
          using std::imag;
          return imag(to_complex(new_colon_separated_string));
        }

        using std::end;
        if (int_variables_.find(variable_name) != end(int_variables_))
          return real_type{0};
        else if (real_variables_.find(variable_name) != end(real_variables_))
          return real_type{0};
        else if (complex_variables_.find(variable_name) != end(complex_variables_))
        {
          using std::imag;
          return imag(to_complex(new_colon_separated_string));
        }
      }

      throw 1;
    }

    auto const found_index = colon_separated_string.find(':');
    if (found_index == std::string::npos)
      return real_variables_.at(colon_separated_string).front();

    using size_type = std::string::size_type;
    return real_variables_.at(colon_separated_string.substr(size_type{0u}, found_index))[to_int(colon_separated_string.substr(found_index + size_type{1u}))];
  }

  auto state::is_complex_symbol(std::string const& symbol_name) const -> bool
  { return symbol_name == ":COMPLEX" or symbol_name == ":I" or symbol_name == ":MINUS_I" or symbol_name == ":RESULT"; }

  auto state::to_complex(std::string const& colon_separated_string) const -> complex_type
  {
    if (colon_separated_string.front() == ':')
    {
      if (colon_separated_string == ":COMPLEX")
        return complex_type{real_type{0}};
      else if (colon_separated_string == ":I")
        return ::ket::utility::imaginary_unit<complex_type>();
      else if (colon_separated_string == ":MINUS_I")
        return ::ket::utility::minus_imaginary_unit<complex_type>();
      else if (colon_separated_string == ":RESULT")
        return result_;

      using size_type = std::string::size_type;
      constexpr auto complex_cast_symbol_length = size_type{9u};
      if (colon_separated_string.size() > complex_cast_symbol_length and colon_separated_string.substr(size_type{0u}, complex_cast_symbol_length) == ":COMPLEX:")
      {
        auto const new_colon_separated_string = colon_separated_string.substr(complex_cast_symbol_length);
        if (new_colon_separated_string.empty())
          throw 1;

        if (std::isdigit(static_cast<unsigned char>(new_colon_separated_string.front()))
            or new_colon_separated_string.front() == '+' or new_colon_separated_string.front() == '-'
            or new_colon_separated_string.front() == '.')
          return static_cast<complex_type>(boost::lexical_cast<real_type>(new_colon_separated_string));

        auto const variable_name
          = new_colon_separated_string.substr(
              size_type{0u},
              new_colon_separated_string.find(
                ':', new_colon_separated_string.front() == ':' ? size_type{1u} : size_type{0u}));

        if (is_int_symbol(variable_name))
          return static_cast<complex_type>(to_int(new_colon_separated_string));
        else if (is_real_symbol(variable_name))
          return static_cast<complex_type>(to_real(new_colon_separated_string));
        else if (is_complex_symbol(variable_name))
          return static_cast<complex_type>(to_complex(new_colon_separated_string));

        using std::end;
        if (int_variables_.find(variable_name) != end(int_variables_))
          return static_cast<complex_type>(to_int(new_colon_separated_string));
        else if (real_variables_.find(variable_name) != end(real_variables_))
          return static_cast<complex_type>(to_real(new_colon_separated_string));
        else if (complex_variables_.find(variable_name) != end(complex_variables_))
          return static_cast<complex_type>(to_complex(new_colon_separated_string));
      }

      throw 1;
    }

    auto const found_index = colon_separated_string.find(':');
    if (found_index == std::string::npos)
      return complex_variables_.at(colon_separated_string).front();

    using size_type = std::string::size_type;
    return complex_variables_.at(colon_separated_string.substr(size_type{0u}, found_index))[to_int(colon_separated_string.substr(found_index + size_type{1u}))];
  }

  auto state::is_pauli_string_space_symbol(std::string const& symbol_name) const -> bool
  { return symbol_name == ":PAULIS"; }

  auto state::to_pauli_string_space(std::string const& colon_separated_string) const -> ::bra::pauli_string_space
  {
    if (colon_separated_string.front() == ':')
    {
      using size_type = std::string::size_type;
      constexpr auto pauli_string_cast_symbol_length = size_type{8u};
      if (colon_separated_string.size() > pauli_string_cast_symbol_length and colon_separated_string.substr(size_type{0u}, pauli_string_cast_symbol_length) == ":PAULIS:")
        return {colon_separated_string.substr(pauli_string_cast_symbol_length), complex_type{real_type{1}}};

      throw 1;
    }

    auto const found_index = colon_separated_string.find(':');
    if (found_index == std::string::npos)
      return pauli_string_space_variables_.at(colon_separated_string).front();

    using size_type = std::string::size_type;
    return pauli_string_space_variables_.at(colon_separated_string.substr(size_type{0u}, found_index))[to_int(colon_separated_string.substr(found_index + size_type{1u}))];
  }

  state& state::i_gate(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_i_gate(qubit);
    return *this;
  }

  state& state::ic_gate(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_ic_gate(control_qubit);
    return *this;
  }

  state& state::ii_gate(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_ii_gate(qubit1, qubit2);
    return *this;
  }

  state& state::in_gate(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_in_gate(qubits);
    return *this;
  }

  state& state::hadamard(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_hadamard(qubit);
    return *this;
  }

  state& state::not_(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_not_(qubit);
    return *this;
  }

  state& state::pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_pauli_x(qubit);
    return *this;
  }

  state& state::pauli_xx(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_pauli_xx(qubit1, qubit2);
    return *this;
  }

  state& state::pauli_xn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_pauli_xn(qubits);
    return *this;
  }

  state& state::pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_pauli_y(qubit);
    return *this;
  }

  state& state::pauli_yy(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_pauli_yy(qubit1, qubit2);
    return *this;
  }

  state& state::pauli_yn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_pauli_yn(qubits);
    return *this;
  }

  state& state::pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_pauli_z(control_qubit);
    return *this;
  }

  state& state::pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_pauli_zz(qubit1, qubit2);
    return *this;
  }

  state& state::pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_pauli_zn(qubits);
    return *this;
  }

  state& state::swap(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_swap(qubit1, qubit2);
    return *this;
  }

  state& state::sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_sqrt_pauli_x(qubit);
    return *this;
  }

  state& state::adj_sqrt_pauli_x(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_sqrt_pauli_x(qubit);
    return *this;
  }

  state& state::sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_sqrt_pauli_y(qubit);
    return *this;
  }

  state& state::adj_sqrt_pauli_y(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_sqrt_pauli_y(qubit);
    return *this;
  }

  state& state::sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_sqrt_pauli_z(control_qubit);
    return *this;
  }

  state& state::adj_sqrt_pauli_z(control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_adj_sqrt_pauli_z(control_qubit);
    return *this;
  }

  state& state::sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_sqrt_pauli_zz(qubit1, qubit2);
    return *this;
  }

  state& state::adj_sqrt_pauli_zz(qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_adj_sqrt_pauli_zz(qubit1, qubit2);
    return *this;
  }

  state& state::sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_sqrt_pauli_zn(qubits);
    return *this;
  }

  state& state::adj_sqrt_pauli_zn(std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_adj_sqrt_pauli_zn(qubits);
    return *this;
  }

  state& state::u1(
    boost::variant<real_type, std::string> const& phase,
    control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubit);
    return *this;
  }

  state& state::adj_u1(
    boost::variant<real_type, std::string> const& phase,
    control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    do_adj_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubit);
    return *this;
  }

  state& state::u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      qubit);
    return *this;
  }

  state& state::adj_u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      qubit);
    return *this;
  }

  state& state::u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      qubit);
    return *this;
  }

  state& state::adj_u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      qubit);
    return *this;
  }

  state& state::phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_phase_shift(phase_coefficients_[phase_exponent_value], control_qubit);
    else
      do_adj_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubit);
    return *this;
  }

  state& state::adj_phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubit);

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_adj_phase_shift(phase_coefficients_[phase_exponent_value], control_qubit);
    else
      do_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubit);
    return *this;
  }

  state& state::x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_x_rotation_half_pi(qubit);
    return *this;
  }

  state& state::adj_x_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_x_rotation_half_pi(qubit);
    return *this;
  }

  state& state::y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_y_rotation_half_pi(qubit);
    return *this;
  }

  state& state::adj_y_rotation_half_pi(qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_y_rotation_half_pi(qubit);
    return *this;
  }

  state& state::exponential_pauli_x(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_exponential_pauli_x(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::adj_exponential_pauli_x(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_exponential_pauli_x(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::exponential_pauli_xx(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_exponential_pauli_xx(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::adj_exponential_pauli_xx(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_adj_exponential_pauli_xx(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::exponential_pauli_xn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_exponential_pauli_xn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::adj_exponential_pauli_xn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_adj_exponential_pauli_xn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::exponential_pauli_y(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_exponential_pauli_y(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::adj_exponential_pauli_y(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubit);

    do_adj_exponential_pauli_y(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::exponential_pauli_yy(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_exponential_pauli_yy(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::adj_exponential_pauli_yy(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_adj_exponential_pauli_yy(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::exponential_pauli_yn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_exponential_pauli_yn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::adj_exponential_pauli_yn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_adj_exponential_pauli_yn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(qubit)], ::bra::found_qubit::ez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(qubit)] = ::bra::found_qubit::ez_qubit;

    do_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::adj_exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit)
  {
    if (is_in_fusion_)
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(qubit)], ::bra::found_qubit::ez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(qubit)] = ::bra::found_qubit::ez_qubit;

    do_adj_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), qubit);
    return *this;
  }

  state& state::exponential_pauli_zz(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_exponential_pauli_zz(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::adj_exponential_pauli_zz(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_adj_exponential_pauli_zz(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::exponential_pauli_zn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_exponential_pauli_zn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::adj_exponential_pauli_zn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, qubits);

    do_adj_exponential_pauli_zn(boost::apply_visitor(real_visitor{*this}, phase), qubits);
    return *this;
  }

  state& state::exponential_swap(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_exponential_swap(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::adj_exponential_swap(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const qubit1, qubit_type const qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, qubit1);
      ::bra::set_found_qubits(found_qubits_, qubit2);
    }

    do_adj_exponential_swap(boost::apply_visitor(real_visitor{*this}, phase), qubit1, qubit2);
    return *this;
  }

  state& state::toffoli(
    qubit_type const target_qubit,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_toffoli(target_qubit, control_qubit1, control_qubit2);
    return *this;
  }

#ifndef BRA_NO_MPI
  state& state::projective_measurement(qubit_type const qubit, yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"M"};

    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit, root);
    last_measured_qubit_ = qubit;
    return *this;
  }

  state& state::measurement(yampi::rank const root, int const precision)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"MEASURE"};

    assert(precision >= 0);

    auto const operation_finish_time = BRA_clock::now(environment_);
    std::ostringstream oss;
    if (communicator_.rank(environment_) == root)
    {
      oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = operation_finish_time;

    do_expectation_values(root);

    if (communicator_.rank(environment_) == root)
    {
      if (precision > 0)
      {
        auto const precision_string = std::to_string(precision);
        auto const width_string = std::to_string(std::max(precision + 3, 8));
        oss.str("");
        oss << fmt::format(std::string{"Expectation values of spins:\n{:^"} + width_string + "s} {:^" + width_string + "s} {:^" + width_string + "s}\n", "<Qx>", "<Qy>", "<Qz>");
        if (maybe_expectation_values_)
          for (auto const& spin: *maybe_expectation_values_)
            oss
              << fmt::format(
                   "{:^ " + width_string + "." + precision_string + "f} {:^ " + width_string + "." + precision_string + "f} {:^ " + width_string + "." + precision_string + "f}\n",
                   0.5 - static_cast<double>(spin[0u]), 0.5 - static_cast<double>(spin[1u]), 0.5 - static_cast<double>(spin[2u]));
        std::cout << oss.str() << std::flush;
      }
      else if (precision < 0)
      {
        oss << "Expectation values of spins:\n<Qx> <Qy> <Qz>\n";
        if (maybe_expectation_values_)
          for (auto const& spin: *maybe_expectation_values_)
            oss << std::setprecision(-precision) << (0.5 - static_cast<double>(spin[0u])) << ' ' << (0.5 - static_cast<double>(spin[1u])) << ' ' << (0.5 - static_cast<double>(spin[2u])) << '\n';
        std::cout << oss.str() << std::flush;
      }
      else
      {
        oss.str("");
        oss << "Expectation values of spins:\n<Qx> <Qy> <Qz>\n";
        if (maybe_expectation_values_)
          for (auto const& spin: *maybe_expectation_values_)
            oss << (0.5 - static_cast<double>(spin[0u])) << ' ' << (0.5 - static_cast<double>(spin[1u])) << ' ' << (0.5 - static_cast<double>(spin[2u])) << '\n';
        std::cout << oss.str() << std::flush;
      }
    }

    auto const expectation_values_finish_time = BRA_clock::now(environment_);
    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Expectation values finished: " << ::bra::state_detail::duration_to_second(start_time_, expectation_values_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, expectation_values_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = expectation_values_finish_time;

    return *this;
  }

  state& state::amplitudes(yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"AMPLITUDES"};

    auto const operation_finish_time = BRA_clock::now(environment_);
    std::ostringstream oss;
    if (communicator_.rank(environment_) == root)
    {
      oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = operation_finish_time;

    do_amplitudes(root);

    auto const amplitudes_finish_time = BRA_clock::now(environment_);
    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Amplitudes finished: " << ::bra::state_detail::duration_to_second(start_time_, amplitudes_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, amplitudes_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = amplitudes_finish_time;

    return *this;
  }

  state& state::generate_events(yampi::rank const root, int const num_events, int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"GENERATE EVENTS"};

    auto const operation_finish_time = BRA_clock::now(environment_);
    std::ostringstream oss;
    if (communicator_.rank(environment_) == root)
    {
      oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = operation_finish_time;

    do_generate_events(root, num_events, seed);

    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Events:\n";
      for (auto index = decltype(num_events){0u}; index < num_events; ++index)
        oss << index << ' ' << ::bra::state_detail::integer_to_bits_string(generated_events_[index], total_num_qubits_) << '\n';
      std::cout << oss.str() << std::flush;
    }

    auto const generate_events_finish_time = BRA_clock::now(environment_);
    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Events finished: " << ::bra::state_detail::duration_to_second(start_time_, generate_events_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, generate_events_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = generate_events_finish_time;

    return *this;
  }

  state& state::exit(yampi::rank const root)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"EXIT"};

    auto const operation_finish_time = BRA_clock::now(environment_);
    std::ostringstream oss;
    if (communicator_.rank(environment_) == root)
    {
      oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = operation_finish_time;

    do_measure(root);

    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Measurement result: " << measured_value_ << '\n';
      std::cout << oss.str() << std::flush;
    }

    auto const measure_finish_time = BRA_clock::now(environment_);
    if (communicator_.rank(environment_) == root)
    {
      oss.str("");
      oss << "Measurement finished: " << ::bra::state_detail::duration_to_second(start_time_, measure_finish_time)
          << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, measure_finish_time) << ")\n";
      std::cout << oss.str() << std::flush;
    }
    last_processed_time_ = measure_finish_time;

    return *this;
  }
#else // BRA_NO_MPI
  state& state::projective_measurement(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"M"};

    last_outcomes_[static_cast<bit_integer_type>(qubit)]
      = do_projective_measurement(qubit);
    last_measured_qubit_ = qubit;
    return *this;
  }

  state& state::measurement(int const precision)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"MEASURE"};

    auto const operation_finish_time = BRA_clock::now();
    std::ostringstream oss;
    oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = operation_finish_time;

    do_expectation_values();

    if (precision > 0)
    {
      auto const precision_string = std::to_string(precision);
      auto const width_string = std::to_string(std::max(precision + 3, 8));
      oss.str("");
      oss << fmt::format(std::string{"Expectation values of spins:\n{:^"} + width_string + "s} {:^" + width_string + "s} {:^" + width_string + "s}\n", "<Qx>", "<Qy>", "<Qz>");
      if (maybe_expectation_values_)
        for (auto const& spin: *maybe_expectation_values_)
          oss
            << fmt::format(
                 "{:^ " + width_string + "." + precision_string + "f} {:^ " + width_string + "." + precision_string + "f} {:^ " + width_string + "." + precision_string + "f}\n",
                 0.5 - static_cast<double>(spin[0u]), 0.5 - static_cast<double>(spin[1u]), 0.5 - static_cast<double>(spin[2u]));
      std::cout << oss.str() << std::flush;
    }
    else if (precision < 0)
    {
      oss << "Expectation values of spins:\n<Qx> <Qy> <Qz>\n";
      if (maybe_expectation_values_)
        for (auto const& spin: *maybe_expectation_values_)
          oss << std::setprecision(-precision) << (0.5 - static_cast<double>(spin[0u])) << ' ' << (0.5 - static_cast<double>(spin[1u])) << ' ' << (0.5 - static_cast<double>(spin[2u])) << '\n';
      std::cout << oss.str() << std::flush;
    }
    else
    {
      oss.str("");
      oss << "Expectation values of spins:\n<Qx> <Qy> <Qz>\n";
      if (maybe_expectation_values_)
        for (auto const& spin: *maybe_expectation_values_)
          oss << (0.5 - static_cast<double>(spin[0u])) << ' ' << (0.5 - static_cast<double>(spin[1u])) << ' ' << (0.5 - static_cast<double>(spin[2u])) << '\n';
      std::cout << oss.str() << std::flush;
    }

    auto const expectation_values_finish_time = BRA_clock::now();
    oss.str("");
    oss << "Expectation values finished: " << ::bra::state_detail::duration_to_second(start_time_, expectation_values_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, expectation_values_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = expectation_values_finish_time;

    return *this;
  }

  state& state::amplitudes()
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"AMPLITUDES"};

    auto const operation_finish_time = BRA_clock::now();
    std::ostringstream oss;
    oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = operation_finish_time;

    do_amplitudes();

    auto const amplitudes_finish_time = BRA_clock::now();
    oss.str("");
    oss << "Amplitudes finished: " << ::bra::state_detail::duration_to_second(start_time_, amplitudes_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, amplitudes_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = amplitudes_finish_time;

    return *this;
  }

  state& state::generate_events(int const num_events, int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"GENERATE EVENTS"};

    auto const operation_finish_time = BRA_clock::now();
    std::ostringstream oss;
    oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = operation_finish_time;

    do_generate_events(num_events, seed);

    oss.str("");
    oss << "Events:\n";
    for (auto index = decltype(num_events){0u}; index < num_events; ++index)
      oss << index << ' ' << ::bra::state_detail::integer_to_bits_string(generated_events_[index], total_num_qubits_) << '\n';
    std::cout << oss.str() << std::flush;

    auto const generate_events_finish_time = BRA_clock::now();
    oss.str("");
    oss << "Events finished: " << ::bra::state_detail::duration_to_second(start_time_, generate_events_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, generate_events_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = generate_events_finish_time;

    return *this;
  }

  state& state::exit()
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"EXIT"};

    auto const operation_finish_time = BRA_clock::now();
    std::ostringstream oss;
    oss << "Operations finished: " << ::bra::state_detail::duration_to_second(start_time_, operation_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, operation_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = operation_finish_time;

    do_measure();

    oss.str("");
    oss << "Measurement result: " << measured_value_ << '\n';
    std::cout << oss.str() << std::flush;

    auto const measure_finish_time = BRA_clock::now();
    oss.str("");
    oss << "Measurement finished: " << ::bra::state_detail::duration_to_second(start_time_, measure_finish_time)
        << " (" << ::bra::state_detail::duration_to_second(last_processed_time_, measure_finish_time) << ")\n";
    std::cout << oss.str() << std::flush;
    last_processed_time_ = measure_finish_time;

    return *this;
  }
#endif // BRA_NO_MPI

  state& state::expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"EXPECTATION VALUE"};

    do_expectation_value(operator_literal_or_variable_name, operated_qubits);

    return *this;
  }

  state& state::shor_box(bit_integer_type const num_exponent_qubits, state_integer_type const divisor, state_integer_type const base)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"SHORBOX"};

    auto exponent_qubits = std::vector<qubit_type>(num_exponent_qubits);
    std::iota(
      std::begin(exponent_qubits), std::end(exponent_qubits),
      static_cast<qubit_type>(total_num_qubits_ - num_exponent_qubits));
    auto modular_exponentiation_qubits
      = std::vector<qubit_type>(total_num_qubits_ - num_exponent_qubits);
    std::iota(
      std::begin(modular_exponentiation_qubits), std::end(modular_exponentiation_qubits),
      qubit_type{0u});

    do_shor_box(divisor, base, exponent_qubits, modular_exponentiation_qubits);

    return *this;
  }

  state& state::begin_fusion()
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"BEGIN FUSION"};

    is_in_fusion_ = true;
    found_qubits_.assign(total_num_qubits_, ::bra::found_qubit::not_found);

    do_begin_fusion();

    return *this;
  }

  state& state::end_fusion()
  {
    if (not is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"END FUSION"};

    do_end_fusion();

    found_qubits_.clear();
    is_in_fusion_ = false;

    return *this;
  }

  state& state::clear(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"CLEAR"};

    do_clear(qubit);
    return *this;
  }

  state& state::set(qubit_type const qubit)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"SET"};

    do_set(qubit);
    return *this;
  }

  state& state::depolarizing_channel(
    real_type const px, real_type const py, real_type const pz,
    int const seed)
  {
    if (is_in_fusion_)
      throw ::bra::unsupported_fused_gate_error{"DEPOLARIZING CHANNEL"};

    using floating_point_type = typename ::bra::utility::closest_floating_point_of<real_type>::type;
    auto distribution = std::uniform_real_distribution<floating_point_type>{0.0, 1.0};
    auto const last_qubit = ket::make_qubit<state_integer_type>(total_num_qubits_);
    if (seed < 0)
      for (auto qubit = ket::make_qubit<state_integer_type>(bit_integer_type{0u}); qubit < last_qubit; ++qubit)
      {
        auto const probability = static_cast<real_type>(distribution(random_number_generator_));
        if (probability < px)
          pauli_x(qubit);
        else if (probability < px + py)
          pauli_y(qubit);
        else if (probability < px + py + pz)
          pauli_z(ket::make_control(qubit));
      }
    else
    {
      auto temporal_random_number_generator = random_number_generator_type{static_cast<seed_type>(seed)};
      for (auto qubit = ket::make_qubit<state_integer_type>(static_cast<bit_integer_type>(0u)); qubit < last_qubit; ++qubit)
      {
        auto const probability = static_cast<real_type>(distribution(temporal_random_number_generator));
        if (probability < px)
          pauli_x(qubit);
        else if (probability < px + py)
          pauli_y(qubit);
        else if (probability < px + py + pz)
          pauli_z(ket::make_control(qubit));
      }
    }

    return *this;
  }

  state& state::controlled_i_gate(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_i_gate(target_qubit, control_qubit);
    return *this;
  }

  state& state::controlled_ic_gate(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_controlled_ic_gate(control_qubit1, control_qubit2);
    return *this;
  }

  state& state::multi_controlled_in_gate(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_in_gate(target_qubits, control_qubits);
    return *this;
  }

  state& state::multi_controlled_ic_gate(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_multi_controlled_ic_gate(control_qubits);
    return *this;
  }

  state& state::controlled_hadamard(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_hadamard(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_hadamard(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_hadamard(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_not(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_not(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_not(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_not(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_pauli_x(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_pauli_xn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_pauli_xn(target_qubits, control_qubits);
    return *this;
  }

  state& state::controlled_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_pauli_y(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_pauli_yn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_pauli_yn(target_qubits, control_qubits);
    return *this;
  }

  state& state::controlled_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_controlled_pauli_z(control_qubit1, control_qubit2);
    return *this;
  }

  state& state::multi_controlled_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_multi_controlled_pauli_z(control_qubits);
    return *this;
  }

  state& state::multi_controlled_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_pauli_zn(target_qubits, control_qubits);
    return *this;
  }

  state& state::multi_controlled_swap(qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit1);
      ::bra::set_found_qubits(found_qubits_, target_qubit2);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_swap(target_qubit1, target_qubit2, control_qubits);
    return *this;
  }

  state& state::controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_sqrt_pauli_x(target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_sqrt_pauli_x(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_sqrt_pauli_x(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_sqrt_pauli_x(target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_sqrt_pauli_x(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_sqrt_pauli_x(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_sqrt_pauli_y(target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_sqrt_pauli_y(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_sqrt_pauli_y(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_sqrt_pauli_y(target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_sqrt_pauli_y(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_sqrt_pauli_y(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_sqrt_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_controlled_sqrt_pauli_z(control_qubit1, control_qubit2);
    return *this;
  }

  state& state::adj_controlled_sqrt_pauli_z(control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_adj_controlled_sqrt_pauli_z(control_qubit1, control_qubit2);
    return *this;
  }

  state& state::multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_multi_controlled_sqrt_pauli_z(control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_sqrt_pauli_z(std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_adj_multi_controlled_sqrt_pauli_z(control_qubits);
    return *this;
  }

  state& state::multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_sqrt_pauli_zn(target_qubits, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_sqrt_pauli_zn(std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_sqrt_pauli_zn(target_qubits, control_qubits);
    return *this;
  }

  state& state::controlled_phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_controlled_phase_shift(phase_coefficients_[phase_exponent_value], control_qubit1, control_qubit2);
    else
      do_adj_controlled_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubit1, control_qubit2);
    return *this;
  }

  state& state::adj_controlled_phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_adj_controlled_phase_shift(phase_coefficients_[phase_exponent_value], control_qubit1, control_qubit2);
    else
      do_controlled_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubit1, control_qubit2);
    return *this;
  }

  state& state::multi_controlled_phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_multi_controlled_phase_shift(phase_coefficients_[phase_exponent_value], control_qubits);
    else
      do_adj_multi_controlled_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_phase_shift(
    boost::variant<int_type, std::string> const& phase_exponent,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    auto const phase_exponent_value = boost::apply_visitor(int_visitor{*this}, phase_exponent);

    if (phase_exponent_value >= 0)
      do_adj_multi_controlled_phase_shift(phase_coefficients_[phase_exponent_value], control_qubits);
    else
      do_multi_controlled_phase_shift(phase_coefficients_[-phase_exponent_value], control_qubits);
    return *this;
  }

  state& state::controlled_u1(
    boost::variant<real_type, std::string> const& phase,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_controlled_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubit1, control_qubit2);
    return *this;
  }

  state& state::adj_controlled_u1(
    boost::variant<real_type, std::string> const& phase,
    control_qubit_type const control_qubit1, control_qubit_type const control_qubit2)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, control_qubit1);
      ::bra::set_found_qubits(found_qubits_, control_qubit2);
    }

    do_adj_controlled_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubit1, control_qubit2);
    return *this;
  }

  state& state::multi_controlled_u1(
    boost::variant<real_type, std::string> const& phase,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_multi_controlled_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_u1(
    boost::variant<real_type, std::string> const& phase,
    std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
      ::bra::set_found_qubits(found_qubits_, control_qubits);

    do_adj_multi_controlled_u1(boost::apply_visitor(real_visitor{*this}, phase), control_qubits);
    return *this;
  }

  state& state::controlled_u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_u2(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_u2(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_u3(
    boost::variant<real_type, std::string> const& phase1,
    boost::variant<real_type, std::string> const& phase2,
    boost::variant<real_type, std::string> const& phase3,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_u3(
      boost::apply_visitor(real_visitor{*this}, phase1),
      boost::apply_visitor(real_visitor{*this}, phase2),
      boost::apply_visitor(real_visitor{*this}, phase3),
      target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_x_rotation_half_pi(target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_x_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_x_rotation_half_pi(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_x_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_x_rotation_half_pi(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_y_rotation_half_pi(target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_y_rotation_half_pi(qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_y_rotation_half_pi(target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_y_rotation_half_pi(qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_y_rotation_half_pi(target_qubit, control_qubits);
    return *this;
  }

  state& state::controlled_exponential_pauli_x(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_exponential_pauli_x(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_exponential_pauli_x(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_exponential_pauli_x(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_exponential_pauli_xn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_exponential_pauli_xn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_exponential_pauli_xn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_exponential_pauli_xn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::controlled_exponential_pauli_y(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_exponential_pauli_y(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_exponential_pauli_y(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit);
      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_exponential_pauli_y(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_exponential_pauli_yn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_exponential_pauli_yn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_exponential_pauli_yn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_exponential_pauli_yn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::controlled_exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)], ::bra::found_qubit::cez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)] = ::bra::found_qubit::cez_qubit;

      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_controlled_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::adj_controlled_exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, control_qubit_type const control_qubit)
  {
    if (is_in_fusion_)
    {
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)], ::bra::found_qubit::cez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)] = ::bra::found_qubit::cez_qubit;

      ::bra::set_found_qubits(found_qubits_, control_qubit);
    }

    do_adj_controlled_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubit);
    return *this;
  }

  state& state::multi_controlled_exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)], ::bra::found_qubit::cez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)] = ::bra::found_qubit::cez_qubit;

      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_exponential_pauli_z(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      if (::bra::is_weaker(found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)], ::bra::found_qubit::cez_qubit))
        found_qubits_[static_cast< ::bra::bit_integer_type >(target_qubit)] = ::bra::found_qubit::cez_qubit;

      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_exponential_pauli_z(boost::apply_visitor(real_visitor{*this}, phase), target_qubit, control_qubits);
    return *this;
  }

  state& state::multi_controlled_exponential_pauli_zn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_exponential_pauli_zn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_exponential_pauli_zn(
    boost::variant<real_type, std::string> const& phase,
    std::vector<qubit_type> const& target_qubits, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubits);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_exponential_pauli_zn(boost::apply_visitor(real_visitor{*this}, phase), target_qubits, control_qubits);
    return *this;
  }

  state& state::multi_controlled_exponential_swap(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit1);
      ::bra::set_found_qubits(found_qubits_, target_qubit2);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_multi_controlled_exponential_swap(boost::apply_visitor(real_visitor{*this}, phase), target_qubit1, target_qubit2, control_qubits);
    return *this;
  }

  state& state::adj_multi_controlled_exponential_swap(
    boost::variant<real_type, std::string> const& phase,
    qubit_type const target_qubit1, qubit_type const target_qubit2, std::vector<control_qubit_type> const& control_qubits)
  {
    if (is_in_fusion_)
    {
      ::bra::set_found_qubits(found_qubits_, target_qubit1);
      ::bra::set_found_qubits(found_qubits_, target_qubit2);
      ::bra::set_found_qubits(found_qubits_, control_qubits);
    }

    do_adj_multi_controlled_exponential_swap(boost::apply_visitor(real_visitor{*this}, phase), target_qubit1, target_qubit2, control_qubits);
    return *this;
  }
} // namespace bra


#undef BRA_clock
