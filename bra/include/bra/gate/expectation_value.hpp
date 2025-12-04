#ifndef BRA_GATE_EXPECTATION_VALUE_HPP
# define BRA_GATE_EXPECTATION_VALUE_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class expectation_value final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string operator_literal_or_variable_name_;
      std::vector<qubit_type> operated_qubits_;

      static std::string const name_;

     public:
      expectation_value(std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits);

      ~expectation_value() = default;
      expectation_value(expectation_value const&) = delete;
      expectation_value& operator=(expectation_value const&) = delete;
      expectation_value(expectation_value&&) = delete;
      expectation_value& operator=(expectation_value&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class expectation_value
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_EXPECTATION_VALUE_HPP
