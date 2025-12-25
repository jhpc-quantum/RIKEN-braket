#ifndef BRA_GATE_INNER_PRODUCT_WITH_OP_HPP
# define BRA_GATE_INNER_PRODUCT_WITH_OP_HPP

# include <string>
# include <vector>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class inner_product_with_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string remote_circuit_index_or_all_;
      std::string operator_literal_or_variable_name_;
      std::vector<qubit_type> operated_qubits_;

      static std::string const name_;

     public:
      inner_product_with_op(std::string const& remote_circuit_index_or_all, std::string const& operator_literal_or_variable_name, std::vector<qubit_type> const& operated_qubits);

      ~inner_product_with_op() = default;
      inner_product_with_op(inner_product_with_op const&) = delete;
      inner_product_with_op& operator=(inner_product_with_op const&) = delete;
      inner_product_with_op(inner_product_with_op&&) = delete;
      inner_product_with_op& operator=(inner_product_with_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class inner_product_with_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_INNER_PRODUCT_WITH_OP_HPP
