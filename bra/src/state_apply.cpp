#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <bra/state.hpp>


namespace bra
{
  namespace
  {
    bool starts_literal(std::string const& operand)
    {
      return not operand.empty() and (std::isdigit(static_cast<unsigned char>(operand.front()))
        or operand.front() == '+' or operand.front() == '-' or operand.front() == '.');
    }

    template <typename T>
    auto apply_element(std::vector<T>& values, int const index, std::string const& operand) -> T*
    {
      if (index < 0 or static_cast<std::size_t>(index) >= values.size())
        throw std::out_of_range{"index out of range in " + operand};
      return &values[static_cast<std::size_t>(index)];
    }

    auto literal_type(std::string const& operand) -> ::bra::variable_type
    {
      auto position = std::size_t{0u};
      if (not operand.empty() and (operand.front() == '+' or operand.front() == '-'))
        ++position;
      auto digits = std::size_t{0u};
      auto consume_digits = [&]
      {
        auto const start = position;
        while (position < operand.size() and std::isdigit(static_cast<unsigned char>(operand[position])))
          ++position;
        return position - start;
      };
      digits += consume_digits();
      auto floating = false;
      if (position < operand.size() and operand[position] == '.')
      {
        floating = true;
        ++position;
        digits += consume_digits();
      }
      if (digits == 0u)
        throw std::invalid_argument{"invalid numeric literal " + operand};
      if (position < operand.size() and (operand[position] == 'E' or operand[position] == 'e'))
      {
        floating = true;
        ++position;
        if (position < operand.size() and (operand[position] == '+' or operand[position] == '-'))
          ++position;
        if (consume_digits() == 0u)
          throw std::invalid_argument{"invalid numeric literal " + operand};
      }
      if (position != operand.size())
        throw std::invalid_argument{"invalid numeric literal " + operand};
      return floating ? ::bra::variable_type::real : ::bra::variable_type::integer;
    }

    auto scalar_real(::bra::classical_function_value const& value) -> ::bra::real_type
    {
      if (auto const pointer = boost::get< ::bra::int_type >(&value)) return static_cast< ::bra::real_type >(*pointer);
      if (auto const pointer = boost::get< ::bra::uint_type >(&value)) return static_cast< ::bra::real_type >(*pointer);
      if (auto const pointer = boost::get< ::bra::real_type >(&value)) return *pointer;
      return boost::get< ::bra::complex_type >(value).real();
    }

    auto cast_scalar(::bra::classical_function_value const& value, ::bra::variable_type const type,
                     bool const imaginary = false) -> ::bra::classical_function_value
    {
      if (imaginary)
      {
        auto const pointer = boost::get< ::bra::complex_type >(&value);
        return pointer == nullptr ? ::bra::real_type{0} : pointer->imag();
      }
      if (type == ::bra::variable_type::complex_)
      {
        if (auto const pointer = boost::get< ::bra::complex_type >(&value)) return *pointer;
        return ::bra::complex_type{scalar_real(value), 0};
      }
      if (type == ::bra::variable_type::real) return scalar_real(value);
      if (type == ::bra::variable_type::integer)
      {
        if (auto const pointer = boost::get< ::bra::int_type >(&value)) return *pointer;
        if (auto const pointer = boost::get< ::bra::uint_type >(&value))
        {
          if (*pointer > static_cast< ::bra::uint_type >(std::numeric_limits< ::bra::int_type >::max()))
            throw std::out_of_range{"UINT to INT conversion out of range"};
          return static_cast< ::bra::int_type >(*pointer);
        }
        auto const real = scalar_real(value);
        auto const limit = std::ldexp(::bra::real_type{1}, std::numeric_limits< ::bra::int_type >::digits);
        if (not boost::math::isfinite(real) or std::trunc(real) < -limit or std::trunc(real) >= limit)
          throw std::out_of_range{"REAL to INT conversion out of range"};
        return static_cast< ::bra::int_type >(real);
      }
      if (auto const pointer = boost::get< ::bra::uint_type >(&value)) return *pointer;
      if (auto const pointer = boost::get< ::bra::int_type >(&value)) return static_cast< ::bra::uint_type >(*pointer);
      auto const real = scalar_real(value);
      auto const limit = std::ldexp(::bra::real_type{1}, std::numeric_limits< ::bra::uint_type >::digits);
      if (not boost::math::isfinite(real) or std::trunc(real) < 0 or std::trunc(real) >= limit)
        throw std::out_of_range{"REAL to UINT conversion out of range"};
      return static_cast< ::bra::uint_type >(real);
    }
  } // namespace

  auto state::apply_variable_type(std::string const& name) const -> ::bra::variable_type
  {
    auto const count = int_variables_.count(name) + uint_variables_.count(name) + real_variables_.count(name)
      + complex_variables_.count(name) + pauli_string_space_variables_.count(name);
    if (count == 0u) throw std::invalid_argument{"unknown variable " + name};
    if (count != 1u) throw std::invalid_argument{"ambiguous variable type for " + name};
    if (int_variables_.count(name)) return ::bra::variable_type::integer;
    if (uint_variables_.count(name)) return ::bra::variable_type::unsigned_integer;
    if (real_variables_.count(name)) return ::bra::variable_type::real;
    if (complex_variables_.count(name)) return ::bra::variable_type::complex_;
    return ::bra::variable_type::pauli_string_space;
  }

  auto state::apply_operand_type(std::string const& operand) const -> ::bra::variable_type
  {
    if (operand.empty()) throw std::invalid_argument{"empty operand"};
    if (starts_literal(operand)) return literal_type(operand);
    auto const name = operand.substr(0u, operand.find(':', operand.front() == ':' ? 1u : 0u));
    if (operand.front() != ':') return apply_variable_type(name);
    if (is_int_symbol(name)) return ::bra::variable_type::integer;
    if (is_uint_symbol(name)) return ::bra::variable_type::unsigned_integer;
    if (is_real_symbol(name)) return ::bra::variable_type::real;
    if (is_complex_symbol(name)) return ::bra::variable_type::complex_;
    if (is_pauli_string_space_symbol(name)) return ::bra::variable_type::pauli_string_space;
    throw std::invalid_argument{"unknown operand symbol " + operand};
  }

  auto state::read_apply_index(std::string const& operand) -> int_type
  {
    auto const value = read_apply_operand(operand);
    if (auto const pointer = boost::get<int_type>(&value)) return *pointer;
    throw std::invalid_argument{"index requires INT or an explicit :INT: cast: " + operand};
  }

  auto state::resolve_apply_destination(std::string const& destination) -> ::bra::classical_function_destination
  {
    if (destination.empty() or not std::isalpha(static_cast<unsigned char>(destination.front())))
      throw std::invalid_argument{"destination must be a mutable variable: " + destination};
    auto const separator = destination.find(':');
    auto const name = destination.substr(0u, separator);
    auto const type = apply_variable_type(name);
    auto const index = separator == std::string::npos ? 0 : read_apply_index(destination.substr(separator + 1u));
    if (index < 0) throw std::out_of_range{"negative index in " + destination};
    switch (type)
    {
    case ::bra::variable_type::integer: return apply_element(int_variables_.at(name), index, destination);
    case ::bra::variable_type::unsigned_integer: return apply_element(uint_variables_.at(name), index, destination);
    case ::bra::variable_type::real: return apply_element(real_variables_.at(name), index, destination);
    case ::bra::variable_type::complex_: return apply_element(complex_variables_.at(name), index, destination);
    default: throw std::invalid_argument{"unsupported destination type: " + destination};
    }
  }

  auto state::read_apply_operand(std::string const& operand) -> ::bra::classical_function_value
  {
    auto const type = apply_operand_type(operand);
    if (starts_literal(operand))
    {
      if (type == ::bra::variable_type::integer) return boost::lexical_cast<int_type>(operand);
      return boost::lexical_cast<real_type>(operand);
    }
    if (operand.front() != ':')
    {
      auto const destination = resolve_apply_destination(operand);
      if (auto const pointer = boost::get<int_type*>(&destination)) return **pointer;
      if (auto const pointer = boost::get<uint_type*>(&destination)) return **pointer;
      if (auto const pointer = boost::get<real_type*>(&destination)) return **pointer;
      return **boost::get<complex_type*>(&destination);
    }
    auto const separator = operand.find(':', 1u);
    auto const name = operand.substr(0u, separator);
    if (separator != std::string::npos)
    {
      auto const argument = operand.substr(separator + 1u);
      if (argument.empty()) throw std::invalid_argument{"empty symbol argument in " + operand};
      if (name == ":OUTCOMES")
      {
        auto const index = read_apply_index(argument);
        if (index < 0) throw std::out_of_range{"negative outcome index"};
        return static_cast<int_type>(last_outcomes_.at(index));
      }
      if (name != ":INT" and name != ":UINT" and name != ":REAL" and name != ":IMAG" and name != ":COMPLEX")
        throw std::invalid_argument{"symbol takes no argument: " + operand};
      // Signed spellings cannot collide with variable identifiers. Capture
      // these REAL representations so the selected function can diagnose
      // nonfinite inputs on execution, including behind classical jumps.
      if (name == ":REAL" and (argument == "+inf" or argument == "-inf" or argument == "+INF" or argument == "-INF"))
        return argument.front() == '-' ? -std::numeric_limits<real_type>::infinity()
                                       : std::numeric_limits<real_type>::infinity();
      if (name == ":REAL" and (argument == "+nan" or argument == "+NAN"))
        return std::numeric_limits<real_type>::quiet_NaN();
      if (starts_literal(argument))
      {
        auto const argument_type = literal_type(argument);
        // Typed numeric forms do not first narrow through the bare INT type.
        if (name == ":UINT" and argument_type == ::bra::variable_type::integer)
        {
          if (argument.front() == '-') return static_cast<uint_type>(boost::lexical_cast<int_type>(argument));
          return boost::lexical_cast<uint_type>(argument);
        }
        if (name == ":INT" and argument_type == ::bra::variable_type::integer)
          return boost::lexical_cast<int_type>(argument);
        return cast_scalar(boost::lexical_cast<real_type>(argument), type, name == ":IMAG");
      }
      return cast_scalar(read_apply_operand(argument), type, name == ":IMAG");
    }
    // Built-in constants without arguments are safe to use through the existing
    // readers. Indexed symbols and casts above use APPLY's bounded resolution.
    if (name == ":OUTCOMES") return static_cast<int_type>(last_outcomes_.at(0u));
    if (name == ":OUTCOME")
      return static_cast<int_type>(last_outcomes_.at(static_cast<bit_integer_type>(last_measured_qubit_)));
    if (type == ::bra::variable_type::integer) return to_int(operand);
    if (type == ::bra::variable_type::unsigned_integer) return to_uint(operand);
    if (type == ::bra::variable_type::real) return to_real(operand);
    if (type == ::bra::variable_type::complex_) return to_complex(operand);
    throw std::invalid_argument{"unsupported operand type: " + operand};
  }

  void state::invoke_apply_operation(
    std::vector<std::string> const& destinations, ::bra::classical_function_id const function,
    std::vector<std::string> const& operands, ::bra::anglebits2phase_chain_ptr phase_chain)
  {
    auto const instruction = ::bra::apply_instruction(destinations, function, operands);
    try
    {
      auto operand_count = operands.size();
      if (function == ::bra::classical_function_id::anglebits2phase)
      {
        if (operand_count < 2u) throw std::invalid_argument{"wrong operand count for ANGLEBITS2PHASE"};
        if (not phase_chain)
          phase_chain = ::bra::parse_anglebits2phase_chain({operands.begin() + 2, operands.end()});
        operand_count = 2u;
      }
      auto locations = std::vector< ::bra::classical_function_destination >{};
      for (auto const& destination: destinations)
        locations.push_back(resolve_apply_destination(destination));
      auto types = std::vector< ::bra::variable_type >{};
      for (auto index = std::size_t{0u}; index < operand_count; ++index)
        types.push_back(apply_operand_type(operands[index]));
      auto const& signature = ::bra::resolve_classical_function(function, types);
      auto values = std::vector< ::bra::classical_function_value >{};
      for (auto index = std::size_t{0u}; index < operand_count; ++index)
        values.push_back(read_apply_operand(operands[index]));
      auto const call = ::bra::classical_function_call{signature, std::move(locations), std::move(values), std::move(phase_chain)};
      call.commit_results(::bra::evaluate_classical_function(call));
    }
    catch (std::exception const& error) { throw ::bra::apply_error{instruction, error.what()}; }
    catch (...) { throw ::bra::apply_error{instruction, "invalid operand or index"}; }
  }
} // namespace bra
