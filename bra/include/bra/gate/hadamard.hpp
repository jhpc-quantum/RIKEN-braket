#ifndef BRA_GATE_HADAMARD_HPP
# define BRA_GATE_HADAMARD_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class hadamard final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;

     private:
      qubit_type qubit_;

      static std::string const name_;

     public:
      explicit hadamard(qubit_type const qubit);

      ~hadamard() = default;
      hadamard(hadamard const&) = delete;
      hadamard& operator=(hadamard const&) = delete;
      hadamard(hadamard&&) = delete;
      hadamard& operator=(hadamard&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class hadamard
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_HADAMARD_HPP
