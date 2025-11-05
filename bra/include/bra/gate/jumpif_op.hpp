#ifndef BRA_GATE_JUMPIF_OP_HPP
# define BRA_GATE_JUMPIF_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class jumpif_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string label_;
      std::string lhs_variable_name_;
      ::bra::compare_operation_type op_;
      std::string rhs_literal_or_variable_name_;

      static std::string const name_;

     public:
      jumpif_op(
        std::string const& label, std::string const& lhs_variable_name,
        ::bra::compare_operation_type const op,
        std::string const& rhs_literal_or_variable_name);

      ~jumpif_op() = default;
      jumpif_op(jumpif_op const&) = delete;
      jumpif_op& operator=(jumpif_op const&) = delete;
      jumpif_op(jumpif_op&&) = delete;
      jumpif_op& operator=(jumpif_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class jumpif_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_JUMPIF_OP_HPP
