#ifndef BRA_CLASSICAL_FUNCTION_CALL_HPP
# define BRA_CLASSICAL_FUNCTION_CALL_HPP

# include <string>
# include <utility>
# include <vector>

# include <boost/variant.hpp>

# include <bra/types.hpp>
# include <bra/classical_function.hpp>
# include <bra/anglebits2phase.hpp>


namespace bra
{
  using classical_function_value = boost::variant< ::bra::int_type, ::bra::uint_type, ::bra::real_type, ::bra::complex_type >;
  using classical_function_destination = boost::variant< ::bra::int_type*, ::bra::uint_type*, ::bra::real_type*, ::bra::complex_type* >;

  class apply_error : public std::runtime_error
  {
   public:
    apply_error(std::string const& instruction, std::string const& reason)
      : std::runtime_error{instruction + ": " + reason} { }
  };

  inline auto apply_instruction(
    std::vector<std::string> const& destinations, ::bra::classical_function_id const function,
    std::vector<std::string> const& operands) -> std::string
  {
    auto instruction = std::string{"APPLY"};
    for (auto const& destination: destinations)
      instruction += " " + destination;
    instruction += " <- " + std::string{classical_function_name(function)};
    for (auto const& operand: operands)
      instruction += " " + operand;
    return instruction;
  }

  namespace classical_function_detail
  {
    inline auto value_type(int const index) -> ::bra::variable_type
    {
      switch (index)
      {
      case 0: return ::bra::variable_type::integer;
      case 1: return ::bra::variable_type::unsigned_integer;
      case 2: return ::bra::variable_type::real;
      case 3: return ::bra::variable_type::complex_;
      }
      throw std::invalid_argument{"invalid classical function value"};
    }

    struct destination_address : boost::static_visitor<void const*>
    {
      template <typename T> void const* operator()(T* const pointer) const { return pointer; }
    };

    struct store_result : boost::static_visitor<>
    {
      ::bra::classical_function_value const& result;
      explicit store_result(::bra::classical_function_value const& value) : result{value} { }
      template <typename T> void operator()(T* const pointer) const { *pointer = boost::get<T>(result); }
    };
  } // namespace classical_function_detail

  // Invocation-local captured values and resolved scalar storage locations.
  // Never retain this object across changes to the state's variable storage.
  class classical_function_call
  {
    ::bra::classical_function_signature signature_;
    std::vector< ::bra::classical_function_destination > destinations_;
    std::vector< ::bra::classical_function_value > operands_;
    ::bra::anglebits2phase_chain_ptr phase_chain_;

   public:
    classical_function_call(
      ::bra::classical_function_signature signature,
      std::vector< ::bra::classical_function_destination > destinations,
      std::vector< ::bra::classical_function_value > operands,
      ::bra::anglebits2phase_chain_ptr phase_chain = {})
      : signature_{std::move(signature)}, destinations_{std::move(destinations)}, operands_{std::move(operands)},
        phase_chain_{std::move(phase_chain)}
    {
      if (phase_chain_ and signature_.function != ::bra::classical_function_id::anglebits2phase)
        throw std::invalid_argument{"unexpected ANGLEBITS2PHASE metadata"};
      if (destinations_.size() != signature_.result_types.size())
        throw std::invalid_argument{"wrong destination count"};
      if (operands_.size() != signature_.operand_types.size())
        throw std::invalid_argument{"wrong operand count"};
      for (auto index = std::size_t{0u}; index < destinations_.size(); ++index)
      {
        if (classical_function_detail::value_type(destinations_[index].which()) != signature_.result_types[index])
          throw std::invalid_argument{"wrong destination type"};
        auto const address = boost::apply_visitor(classical_function_detail::destination_address{}, destinations_[index]);
        if (address == nullptr)
          throw std::invalid_argument{"null destination"};
        for (auto previous = std::size_t{0u}; previous < index; ++previous)
          if (address == boost::apply_visitor(classical_function_detail::destination_address{}, destinations_[previous]))
            throw std::invalid_argument{"duplicate destination location"};
      }
      for (auto index = std::size_t{0u}; index < operands_.size(); ++index)
        if (classical_function_detail::value_type(operands_[index].which()) != signature_.operand_types[index])
          throw std::invalid_argument{"wrong operand type"};
    }

    auto signature() const -> ::bra::classical_function_signature const& { return signature_; }
    auto operands() const -> std::vector< ::bra::classical_function_value > const& { return operands_; }
    auto phase_chain() const -> ::bra::anglebits2phase_chain_ptr const& { return phase_chain_; }

    void commit_results(std::vector< ::bra::classical_function_value > const& results) const
    {
      if (results.size() != destinations_.size())
        throw std::invalid_argument{"wrong result count"};
      // Check every result before writing. Assigning a matching built-in scalar
      // or std::complex value cannot throw after this validation.
      for (auto index = std::size_t{0u}; index < results.size(); ++index)
        if (classical_function_detail::value_type(results[index].which()) != signature_.result_types[index])
          throw std::invalid_argument{"wrong result type"};
      for (auto index = std::size_t{0u}; index < results.size(); ++index)
        boost::apply_visitor(classical_function_detail::store_result{results[index]}, destinations_[index]);
    }
  };

  auto evaluate_classical_function(::bra::classical_function_call const& call)
    -> std::vector< ::bra::classical_function_value >;
} // namespace bra


#endif // BRA_CLASSICAL_FUNCTION_CALL_HPP
