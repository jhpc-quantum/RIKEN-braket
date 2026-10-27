#ifndef BRA_GATE_APPLY_OP_HPP
# define BRA_GATE_APPLY_OP_HPP

# include <string>
# include <vector>

# include <bra/classical_function.hpp>
# include <bra/anglebits2phase.hpp>
# include <bra/gate/gate.hpp>


namespace bra
{
  namespace gate
  {
    class apply_op final : public ::bra::gate::gate
    {
      std::vector<std::string> destinations_;
      ::bra::classical_function_id function_;
      std::vector<std::string> operands_;
      ::bra::anglebits2phase_chain_ptr phase_chain_;
      static std::string const name_;

     public:
      apply_op(std::vector<std::string> destinations, ::bra::classical_function_id function,
               std::vector<std::string> operands);

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(std::ostringstream& stream, int parameter_width) const override;
    };
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_APPLY_OP_HPP
