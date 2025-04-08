#ifndef BRA_GATE_CONTROLLED_U1_HPP
# define BRA_GATE_CONTROLLED_U1_HPP

# include <string>
# include <iosfwd>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class controlled_u1 final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      real_type phase_;
      control_qubit_type control_qubit1_;
      control_qubit_type control_qubit2_;

      static std::string const name_;

     public:
      controlled_u1(
        real_type const phase,
        control_qubit_type const control_qubit1,
        control_qubit_type const control_qubit2);

      ~controlled_u1() = default;
      controlled_u1(controlled_u1 const&) = delete;
      controlled_u1& operator=(controlled_u1 const&) = delete;
      controlled_u1(controlled_u1&&) = delete;
      controlled_u1& operator=(controlled_u1&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class controlled_u1
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_CONTROLLED_U1_HPP
