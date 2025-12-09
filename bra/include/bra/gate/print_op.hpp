#ifndef BRA_GATE_PRINT_OP_HPP
# define BRA_GATE_PRINT_OP_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class print_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<std::string> variables_or_literals_;

      static std::string const name_;

     public:
      explicit print_op(std::vector<std::string> const& variables_or_literals);
      explicit print_op(std::vector<std::string>&& variables_or_literals);

      ~print_op() = default;
      print_op(print_op const&) = delete;
      print_op& operator=(print_op const&) = delete;
      print_op(print_op&&) = delete;
      print_op& operator=(print_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class print_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PRINT_OP_HPP
