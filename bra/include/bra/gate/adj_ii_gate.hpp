#ifndef BRA_GATE_ADJ_II_GATE_HPP
# define BRA_GATE_ADJ_II_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_ii_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      adj_ii_gate(qubit_type const qubit1, qubit_type const qubit2);

      ~adj_ii_gate() = default;
      adj_ii_gate(adj_ii_gate const&) = delete;
      adj_ii_gate& operator=(adj_ii_gate const&) = delete;
      adj_ii_gate(adj_ii_gate&&) = delete;
      adj_ii_gate& operator=(adj_ii_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_ii_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_II_GATE_HPP
