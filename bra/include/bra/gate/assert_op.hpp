#ifndef BRA_GATE_ASSERT_OP_HPP
# define BRA_GATE_ASSERT_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>

namespace bra
{
  namespace gate
  {
    class assert_op final
      : public ::bra::gate::gate
    {
      std::string lhs_variable_name_;
      ::bra::compare_operation_type op_;
      std::string rhs_literal_or_variable_name_;
      static std::string const name_;

     public:
      assert_op(std::string const& lhs_variable_name, ::bra::compare_operation_type const op,
                std::string const& rhs_literal_or_variable_name);
      ~assert_op() = default;
      assert_op(assert_op const&) = delete;
      assert_op& operator=(assert_op const&) = delete;
      assert_op(assert_op&&) = delete;
      assert_op& operator=(assert_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class assert_op
  } // namespace gate
} // namespace bra

#endif // BRA_GATE_ASSERT_OP_HPP
