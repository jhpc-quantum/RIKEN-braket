#ifndef BRA_GATE_U1_HPP
# define BRA_GATE_U1_HPP

# include <string>
# include <iosfwd>

# include <boost/variant/variant.hpp>

# include <bra/gate/gate.hpp>
# include <bra/state.hpp>


namespace bra
{
  namespace gate
  {
    class u1 final
      : public ::bra::gate::gate
    {
     public:
      using control_qubit_type = ::bra::state::control_qubit_type;
      using real_type = ::bra::state::real_type;

     private:
      boost::variant<real_type, std::string> phase_;
      control_qubit_type control_qubit_;

      static std::string const name_;

     public:
      u1(boost::variant<real_type, std::string> const& phase, control_qubit_type const control_qubit);

      ~u1() = default;
      u1(u1 const&) = delete;
      u1& operator=(u1 const&) = delete;
      u1(u1&&) = delete;
      u1& operator=(u1&&) = delete;

     private:
      ::bra::state& do_apply(::bra::state& state) const override;
      std::string const& do_name() const override;
      std::string do_representation(
        std::ostringstream& repr_stream, int const parameter_width) const override;
    }; // class u1
  } // namespace gate
} // namespace bra


#endif // BRA_GATE_U1_HPP
