#ifndef BRA_GATE_II_GATE_HPP
# define BRA_GATE_II_GATE_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class ii_gate final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit1_;
      qubit_type qubit2_;

      static std::string const name_;

     public:
      ii_gate(qubit_type const qubit1, qubit_type const qubit2);

      ~ii_gate() = default;
      ii_gate(ii_gate const&) = delete;
      ii_gate& operator=(ii_gate const&) = delete;
      ii_gate(ii_gate&&) = delete;
      ii_gate& operator=(ii_gate&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class ii_gate
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_II_GATE_HPP
