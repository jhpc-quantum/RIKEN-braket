#ifndef BRA_GATE_SCATTER_OP_HPP
# define BRA_GATE_SCATTER_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class scatter_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      int root_circuit_index_;
      std::string variable_name_;
      ::bra::variable_type type_;
      int num_elements_;
      std::string source_variable_name_;

      static std::string const name_;

     public:
      scatter_op(int const root_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements, std::string const& source_variable_name);

      ~scatter_op() = default;
      scatter_op(scatter_op const&) = delete;
      scatter_op& operator=(scatter_op const&) = delete;
      scatter_op(scatter_op&&) = delete;
      scatter_op& operator=(scatter_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class scatter_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SCATTER_OP_HPP
