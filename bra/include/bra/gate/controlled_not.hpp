#ifndef BRA_GATE_CONTROLLED_NOT_HPP
# define BRA_GATE_CONTROLLED_NOT_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_not final
      : public ::bra::gate::gate
    {
     public:
      using qubit_type = ::bra::state::qubit_type;
      using control_qubit_type = ::bra::state::control_qubit_type;

     private:
      qubit_type target_qubit_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      controlled_not(
        qubit_type const target_qubit,
        control_qubit_type const control_qubit);

      ~controlled_not() = default;
      controlled_not& operator=(controlled_not const&) = delete;
      controlled_not(controlled_not&&) = delete;
      controlled_not& operator=(controlled_not&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_not
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_NOT_HPP
