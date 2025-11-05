#ifndef BRA_GATE_JUMP_OP_HPP
# define BRA_GATE_JUMP_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class jump_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      std::string label_;

      static std::string const name_;

     public:
      explicit jump_op(std::string const& label);

      ~jump_op() = default;
      jump_op(jump_op const&) = delete;
      jump_op& operator=(jump_op const&) = delete;
      jump_op(jump_op&&) = delete;
      jump_op& operator=(jump_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class jump_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_JUMP_OP_HPP
