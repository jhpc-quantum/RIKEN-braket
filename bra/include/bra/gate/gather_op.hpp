#ifndef BRA_GATE_GATHER_OP_HPP
# define BRA_GATE_GATHER_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class gather_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      int root_circuit_index_;
      std::string variable_name_;
      ::bra::variable_type type_;
      int num_elements_;
      std::string destination_variable_name_;

      static std::string const name_;

     public:
      gather_op(int const root_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements, std::string const& destination_variable_name);

      ~gather_op() = default;
      gather_op(gather_op const&) = delete;
      gather_op& operator=(gather_op const&) = delete;
      gather_op(gather_op&&) = delete;
      gather_op& operator=(gather_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class gather_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_GATHER_OP_HPP
