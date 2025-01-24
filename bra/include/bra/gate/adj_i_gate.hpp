#ifndef BRA_GATE_ADJ_I_GATE_HPP
# define BRA_GATE_ADJ_I_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_i_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_i_gate(qubit_type const qubit);

      ~adj_i_gate() = default;
      adj_i_gate(adj_i_gate const&) = delete;
      adj_i_gate& operator=(adj_i_gate const&) = delete;
      adj_i_gate(adj_i_gate&&) = delete;
      adj_i_gate& operator=(adj_i_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_i_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_I_GATE_HPP
