#ifndef BRA_GATE_PRINTLN_OP_HPP
# define BRA_GATE_PRINTLN_OP_HPP

# include <vector>
# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class println_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::vector<std::string> variables_or_literals_;

      static std::string const name_;

     public:
      explicit println_op(std::vector<std::string> const& variables_or_literals);
      explicit println_op(std::vector<std::string>&& variables_or_literals);

      ~println_op() = default;
      println_op(println_op const&) = delete;
      println_op& operator=(println_op const&) = delete;
      println_op(println_op&&) = delete;
      println_op& operator=(println_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class println_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_PRINTLN_OP_HPP
