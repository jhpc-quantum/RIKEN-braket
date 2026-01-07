#ifndef BRA_GATE_SEND_OP_HPP
# define BRA_GATE_SEND_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class send_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      int destination_circuit_index_;
      std::string variable_name_;
      ::bra::variable_type type_;
      int num_elements_;

      static std::string const name_;

     public:
      send_op(int const destination_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements);

      ~send_op() = default;
      send_op(send_op const&) = delete;
      send_op& operator=(send_op const&) = delete;
      send_op(send_op&&) = delete;
      send_op& operator=(send_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class send_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_SEND_OP_HPP
