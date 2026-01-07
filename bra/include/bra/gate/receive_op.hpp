#ifndef BRA_GATE_RECEIVE_OP_HPP
# define BRA_GATE_RECEIVE_OP_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class receive_op final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      int source_circuit_index_;
      std::string variable_name_;
      ::bra::variable_type type_;
      int num_elements_;

      static std::string const name_;

     public:
      receive_op(int const source_circuit_index, std::string const& variable_name, ::bra::variable_type const type, int const num_elements);

      ~receive_op() = default;
      receive_op(receive_op const&) = delete;
      receive_op& operator=(receive_op const&) = delete;
      receive_op(receive_op&&) = delete;
      receive_op& operator=(receive_op&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class receive_op
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_RECEIVE_OP_HPP
