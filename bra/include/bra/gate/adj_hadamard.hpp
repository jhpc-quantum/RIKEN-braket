#ifndef BRA_GATE_ADJ_HADAMARD_HPP
# define BRA_GATE_ADJ_HADAMARD_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class adj_hadamard final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit adj_hadamard(qubit_type const qubit);

      ~adj_hadamard() = default;
      adj_hadamard(adj_hadamard const&) = delete;
      adj_hadamard& operator=(adj_hadamard const&) = delete;
      adj_hadamard(adj_hadamard&&) = delete;
      adj_hadamard& operator=(adj_hadamard&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class adj_hadamard
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_ADJ_HADAMARD_HPP
