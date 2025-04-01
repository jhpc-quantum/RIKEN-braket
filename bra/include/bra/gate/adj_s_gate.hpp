#ifndef BRA_GATE_ADJ_S_GATE_HPP
# define BRA_GATE_ADJ_S_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_s_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_s_gate(qubit_type const qubit);

      ~adj_s_gate() = default;
      adj_s_gate(adj_s_gate const&) = delete;
      adj_s_gate& operator=(adj_s_gate const&) = delete;
      adj_s_gate(adj_s_gate&&) = delete;
      adj_s_gate& operator=(adj_s_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_s_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_S_GATE_HPP
