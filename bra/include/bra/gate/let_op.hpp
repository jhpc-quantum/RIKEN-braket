#ifndef BRA_GATE_LET_OP_HPP
# define BRA_GATE_LET_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class let_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string lhs_variable_name_;
      ::bra::assign_operation_type op_;
      std::string rhs_literal_or_variable_name_;

      static std::string const name_;

     public:
      let_op(std::string const& lhs_variable_name, ::bra::assign_operation_type const op, std::string const& rhs_literal_or_variable_name);

      ~let_op() = default;
      let_op(let_op const&) = delete;
      let_op& operator=(let_op const&) = delete;
      let_op(let_op&&) = delete;
      let_op& operator=(let_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class let_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_LET_OP_HPP
