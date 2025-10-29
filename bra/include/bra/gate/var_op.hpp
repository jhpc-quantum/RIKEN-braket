#ifndef BRA_GATE_VAR_OP_HPP
# define BRA_GATE_VAR_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class var_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string variable_name_;
      ::bra::variable_type type_;
      int num_elements_;

      static std::string const name_;

     public:
      var_op(std::string const& variable_name, ::bra::variable_type const type, int const num_elements);

      ~var_op() = default;
      var_op(var_op const&) = delete;
      var_op& operator=(var_op const&) = delete;
      var_op(var_op&&) = delete;
      var_op& operator=(var_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class var_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_VAR_OP_HPP
